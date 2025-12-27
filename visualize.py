
# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
from utils import imwrite # 显式导入用于保存图像的函数

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def main(conf: conf_mgt.Default_Conf, save_every=10):
    """
    主函数: 执行图像生成可视化
    :param conf:配置对象，包含模型和运行参数
    :param save_every: 每隔多少步保存中间结果
    """

    print("Start Visualization", conf['name'])

    device = dist_util.dev(conf.get('device'))

    # 创建模型和扩散过程
    # create_model_and_diffusion: 根据配置初始化UNet模型和高斯扩散过程
    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        # 如果启用了分类器引导(classifier guidance)，则加载分类器模型
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            # 定义引导函数：计算分类器梯度以引导生成过程
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                # 返回特定类别的梯度 * scale
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        # 模型前向传播函数
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")

    dset = 'eval'
    eval_name = conf.get_default_eval_name()
    # 获取数据加载器
    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    for batch in iter(dl):
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}
        model_kwargs["gt"] = batch['GT']

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes

        # 选择逐步采样函数 (Progressive Sampling)
        if conf.use_ddim:
            if hasattr(diffusion, 'ddim_sample_loop_progressive'):
                sample_fn_progressive = diffusion.ddim_sample_loop_progressive
            else:
                print("Warning: DDIM progressive sampling not found, falling back to P-Sample progressive.")
                sample_fn_progressive = diffusion.p_sample_loop_progressive
        else:
            sample_fn_progressive = diffusion.p_sample_loop_progressive

        # 逐步生成样本
        generator = sample_fn_progressive(
            model_fn,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            conf=conf
        )

        # 中间结果输出目录
        out_base_path = conf['data'][dset][eval_name]['paths']['srs']
        intermediate_path = os.path.join(out_base_path, "intermediate")
        os.makedirs(intermediate_path, exist_ok=True)

        img_name_base = batch['GT_name'][0].split('.')[0] # 假设batch size为1，简化命名

        print(f"Generating for {img_name_base}...")

        step = 0
        for result in generator:
            step += 1
            if step % save_every == 0 or step == 1:
                sample = result['sample'] # This is x_t
                # You can also use result['pred_xstart'] for the denoised prediction

                srs = toU8(sample)

                # 保存中间步骤的图像
                # 假设batch size为1
                img_out_name = f"{img_name_base}_step_{step:04d}.png"
                full_path = os.path.join(intermediate_path, img_out_name)

                # imwrite 期望 [H, W, C] 格式的 numpy uint8 数组，toU8 函数已经转换好了
                imwrite(img=srs[0], path=full_path)

        # 保存最终结果，与原始脚本一致
        srs = toU8(result['sample'])
        gts = toU8(result['gt'])
        # 计算低分辨率或被遮挡的图像 (LRS)
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))
        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

        # 调用 conf.eval_imswrite 保存所有相关图像 (GT, Mask, Input, Output)
        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

    print("Sampling and visualization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None, help='配置文件路径')
    parser.add_argument('--save_every', type=int, default=10, help="每N步保存一次中间图像")
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg, save_every=args.get('save_every'))
