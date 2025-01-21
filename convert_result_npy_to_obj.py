import os.path as osp

import numpy as np
import torch

import core.utils as utils
from core import get_recon_model
from core.options import ImageFittingOptions


def init_model(args):
    # init face detection and lms detection models
    print("loading models")

    recon_model = get_recon_model(
        model=args.recon_model,
        device=args.device,
        batch_size=1,
        img_size=args.tar_size,
    )

    return recon_model


def convert_npy_to_obj(recon_model, npy_path, res_folder=None):
    # root, ext = osp.splitext(npy_path))

    coeffs = np.load(npy_path)
    coeffs = torch.tensor(
        coeffs,
        dtype=torch.float32,
        device=recon_model.device,
    )

    print(f"--> coeffs.shape: {coeffs.shape}")
    coeffs = coeffs.unsqueeze(0)

    with torch.no_grad():
        # coeffs = recon_model.get_packed_tensors()
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation = (
            recon_model.split_coeffs(coeffs)
        )

        utils.mymkdirs(res_folder)
        basename = osp.splitext(osp.basename(npy_path))[0]

        # save the mesh into obj format
        out_obj_path = osp.join(
            res_folder,
            basename + "_mesh_no_illum.obj",
        )
        # vs = pred_dict["vs"].cpu().numpy().squeeze()
        # tri = pred_dict["tri"].cpu().numpy().squeeze()
        # color = pred_dict["color"].cpu().numpy().squeeze()
        vs = recon_model.get_vs(id_coeff, exp_coeff)
        vs_npy = vs.cpu().numpy().squeeze()
        tri_npy = recon_model.tri.cpu().numpy().squeeze()
        face_texture = recon_model.get_color(tex_coeff)
        color_npy = face_texture.cpu().numpy().squeeze() / 255.0
        utils.save_obj(out_obj_path, vs_npy, tri_npy + 1, color_npy)

        print(f"--> Mesh with texture and illumination saved to {out_obj_path}")

        out_obj_path = osp.join(
            res_folder,
            basename + "_mesh_with_illum.obj",
        )
        rotation = recon_model.compute_rotation_matrix(angles)
        face_norm = recon_model.compute_norm(
            vs,
            recon_model.tri,
            recon_model.point_buf,
        )
        face_norm_r = face_norm.bmm(rotation)
        face_color = recon_model.add_illumination(face_texture, face_norm_r, gamma)
        color_npy = face_color.cpu().numpy().squeeze() / 255.0
        utils.save_obj(out_obj_path, vs_npy, tri_npy + 1, color_npy)

        print(f"--> Mesh with texture and illumination saved to {out_obj_path}")


if __name__ == "__main__":
    args = ImageFittingOptions()
    args = args.parse()
    # args.device = 'cuda:%d' % args.gpu
    args.device = "mps"

    recon_model = init_model(args)
    convert_npy_to_obj(recon_model, args.img_path, args.res_folder)
