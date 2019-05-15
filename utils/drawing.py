#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

import string

import numpy as np
from habitat.utils.visualizations import maps
import cv2

BORDER = 0
CV_FONT = cv2.FONT_HERSHEY_DUPLEX
FANCY_FONT = None
FONT_SIZE = 18


def subplot(
    plots,
    rows,
    cols,
    output_width,
    output_height,
    border=BORDER,
    titles=None,
    normalize=None,
    order=None,
    fancy_text=False,
):
    """
    Given a list of images, returns a single image with the sub-images tiled in a subplot.

    :param plots: array of numpy array images to plot. Can be of different sizes and dimensions as
        long as they are 2 or 3 dimensional.
    :param rows: int number of rows in subplot. If there are fewer images than rows, it will add
        empty space for the blanks. If there are fewer rows than images, it will not draw the
        remaining images.
    :param cols: int number of columns in subplot. Similar to rows.
    :param output_width: int width in pixels of a single subplot output image.
    :param output_height: int height in pixels of a single subplot output image.
    :param border: int amount of border padding pixels between each image.
    :param titles: titles for each subplot to be rendered on top of images.
    :param normalize: list of whether to subtract the max and divide by the min before colorizing.
        If none, assumed false for all images.
    :param order: if provided this reorders the provided plots before drawing them.
    :param fancy_text: if true, uses a fancier font than CV_FONT, but takes longer to render.
    :return: A single image containing the provided images (up to rows * cols).
    """
    global FANCY_FONT
    global FONT_SIZE

    if order is not None:
        plots = [plots[im_ind] for im_ind in order]
        if titles is not None:
            titles = [titles[im_ind] for im_ind in order]
        if normalize is not None:
            normalize = [normalize[im_ind] for im_ind in order]

    returned_image = np.full(
        ((output_height + 2 * border) * rows, (output_width + 2 * border) * cols, 3), 191, dtype=np.uint8
    )
    if fancy_text:
        from PIL import Image, ImageDraw, ImageFont

        if FANCY_FONT is None:
            FONT_SIZE = int(FONT_SIZE * output_width / 320.0)
            FANCY_FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SIZE)
    for row in range(rows):
        for col in range(cols):
            try:
                if col + cols * row >= len(plots):
                    return returned_image
                im = plots[col + cols * row]
                if im is None:
                    continue
                im = im.squeeze()
                if im.dtype != np.uint8 or len(im.shape) < 3:
                    if normalize is None or normalize[col + cols * row]:
                        im = im.astype(np.float32)
                        im -= np.min(im)
                        im *= 255 / max(np.max(im), 1e-10)
                        im = im.astype(np.uint8)
                    else:
                        im = im.astype(np.uint8)
                if len(im.shape) < 3:
                    im = cv2.applyColorMap(im, cv2.COLORMAP_JET)[:, :, ::-1]
                if im.shape != (output_height, output_width, 3):
                    im_width = im.shape[1] * output_height / im.shape[0]
                    if im_width > output_width:
                        im_width = output_width
                        im_height = im.shape[0] * output_width / im.shape[1]
                    else:
                        im_width = im.shape[1] * output_height / im.shape[0]
                        im_height = output_height
                    im_width = int(im_width)
                    im_height = int(im_height)
                    im = cv2.resize(im, (im_width, im_height), interpolation=cv2.INTER_NEAREST)
                    if im_width != output_width:
                        pad0 = int(np.floor((output_width - im_width) * 1.0 / 2))
                        pad1 = int(np.ceil((output_width - im_width) * 1.0 / 2))
                        im = np.lib.pad(im, ((0, 0), (pad0, pad1), (0, 0)), "constant", constant_values=0)
                    elif im_height != output_height:
                        pad0 = int(np.floor((output_height - im_height) * 1.0 / 2))
                        pad1 = int(np.ceil((output_height - im_height) * 1.0 / 2))
                        im = np.lib.pad(im, ((pad0, pad1), (0, 0), (0, 0)), "constant", constant_values=0)
                if (
                    titles is not None
                    and len(titles) > 1
                    and len(titles) > col + cols * row
                    and len(titles[col + cols * row]) > 0
                ):
                    if fancy_text:
                        if im.dtype != np.uint8:
                            im = im.astype(np.uint8)
                        im = Image.fromarray(im)
                        draw = ImageDraw.Draw(im)
                        if isinstance(titles[col + cols * row], str):
                            for x in range(6, 9):
                                for y in range(3, 6):
                                    draw.text((x, y), titles[col + cols * row], (0, 0, 0), font=FANCY_FONT)
                            draw.text((7, 4), titles[col + cols * row], (255, 255, 255), font=FANCY_FONT)
                        else:
                            for tt, title in enumerate(titles[col + cols * row]):
                                for x in range(6, 9):
                                    for y in range(3, 6):
                                        draw.text((x, y + tt * (FONT_SIZE + 5)), title, (0, 0, 0), font=FANCY_FONT)
                                draw.text((7, 4 + tt * (FONT_SIZE + 5)), title, (255, 255, 255), font=FANCY_FONT)
                        im = np.asarray(im)
                    else:
                        scale_factor = im.shape[1] / 320.0
                        if isinstance(titles[col + cols * row], str):
                            im = cv2.putText(
                                im.copy(),
                                titles[col + cols * row],
                                (30, int(30 * scale_factor)),
                                CV_FONT,
                                0.5 * scale_factor,
                                [0, 0, 0],
                                4,
                            )
                            im = cv2.putText(
                                im.copy(),
                                titles[col + cols * row],
                                (30, int(30 * scale_factor)),
                                CV_FONT,
                                0.5 * scale_factor,
                                [255, 255, 255],
                                1,
                            )
                        else:
                            for tt, title in enumerate(titles[col + cols * row]):
                                im = cv2.putText(
                                    im.copy(),
                                    title,
                                    (30, int((tt + 1) * 30 * scale_factor)),
                                    CV_FONT,
                                    0.5 * scale_factor,
                                    [0, 0, 0],
                                    4,
                                )
                                im = cv2.putText(
                                    im.copy(),
                                    title,
                                    (30, int((tt + 1) * 30 * scale_factor)),
                                    CV_FONT,
                                    0.5 * scale_factor,
                                    [255, 255, 255],
                                    1,
                                )
                returned_image[
                    border + (output_height + border) * row : (output_height + border) * (row + 1),
                    border + (output_width + border) * col : (output_width + border) * (col + 1),
                    :,
                ] = im
            except Exception as ex:
                import traceback

                traceback.print_exc()
                print("Failed for image", col + cols * row)
                print("shape", plots[col + cols * row].shape)
                print("type", plots[col + cols * row].dtype)
                if titles is not None and len(titles) > col + col * row:
                    print("title", titles[col + cols * row])
                import pdb

                pdb.set_trace()
                print("bad")
                raise ex

    im = returned_image
    # for one long title
    if titles is not None and len(titles) == 1:
        if fancy_text:
            if im.dtype != np.uint8:
                im = im.astype(np.uint8)
            im = Image.fromarray(im)
            draw = ImageDraw.Draw(im)
            for x in range(9, 12):
                for y in range(9, 12):
                    draw.text((x, y), titles[0], (0, 0, 0), font=FANCY_FONT)
            draw.text((10, 10), titles[0], (255, 255, 255), font=FANCY_FONT)
            im = np.asarray(im)
        else:
            scale_factor = max(max(im.shape[0], im.shape[1]) * 1.0 / 300, 1)
            cv2.putText(im, titles[0], (10, 30), CV_FONT, 0.5 * scale_factor, [0, 0, 0], 4)
            cv2.putText(im, titles[0], (10, 30), CV_FONT, 0.5 * scale_factor, [255, 255, 255], 1)

    return im


def draw_probability_hist(pi):
    p_size = max(len(pi), 100)
    p_hist = np.zeros((p_size, p_size), dtype=np.int32)
    for ii, pi_i in enumerate(pi):
        p_hist[
            : np.clip(int(np.round(pi_i * p_size)), 1, 99),
            int(ii * p_size / len(pi)) : int((ii + 1) * p_size / len(pi)),
        ] = (ii + 1)
    p_hist = np.flipud(p_hist)
    return p_hist


def obs_to_images(obs):
    img = obs["rgb"].copy()
    images = [img.transpose(0, 2, 3, 1)]

    # Draw top down view
    if "visited_grid" in obs:
        top_down_map = obs["visited_grid"][0, ...]
    elif "top_down_map" in obs:
        top_down_map = maps.colorize_topdown_map(obs["top_down_map"]["map"])
        map_size = 1024
        original_map_size = top_down_map.shape[:2]
        if original_map_size[0] > original_map_size[1]:
            map_scale = np.array((1, original_map_size[1] * 1.0 / original_map_size[0]))
        else:
            map_scale = np.array((original_map_size[0] * 1.0 / original_map_size[1], 1))

        new_map_size = np.round(map_size * map_scale).astype(np.int32)
        # OpenCV expects w, h but map size is in h, w
        top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

        map_agent_pos = obs["top_down_map"]["agent_map_coord"]
        map_agent_pos = np.round(map_agent_pos * new_map_size / original_map_size).astype(np.int32)
        top_down_map = maps.draw_agent(
            top_down_map, map_agent_pos, obs["heading"] - np.pi / 2, agent_radius_px=top_down_map.shape[0] / 40
        )
    else:
        top_down_map = None

    normalize = [True]
    titles = [
        (
            ("Method: %s" % obs["method"].replace("_", " ")),
            ("Step: %03d Reward: %.3f" % (obs["step"][0], obs.get("reward", [0])[0])),
            ("Action: %s" % string.capwords(obs["action_taken_name"].replace("_", " "))),
        )
    ]
    images.append(top_down_map)
    if "visited" in obs:
        titles.append((("Visited Cube Count:  %d" % obs["visited"][0]),))
    elif "distance_from_start" in obs:
        titles.append("Geo Dist From Origin: %.3f" % obs["distance_from_start"][0])
    elif "pointgoal" in obs:
        titles.append(
            (("Euc Dist:  %.3f" % obs["pointgoal"][0, 0]), ("Geo Dist: %.3f" % obs["goal_geodesic_distance"][0]))
        )

    normalize.append(False)

    for key, val in obs.items():
        if key == "depth" or key == "output_depth":
            normalize.append(False)
            val = val[:, 0, ...]
            depth = np.clip(val, -0.5, 0.5)
            depth += 0.5
            depth *= 255
            titles.append(key)
            depth = depth.astype(np.uint8)
            depth = np.reshape(depth, (-1, depth.shape[-1]))
            images.append(depth)
        elif key == "surface_normals" or key == "output_surface_normals":
            titles.append(key)
            normalize.append(False)
            val = val.copy()
            if key == "output_surface_normals":
                # Still need to be normalized
                val /= np.sqrt(np.sum(val ** 2, axis=1, keepdims=True))
            surfnorm = (np.clip((val + 1), 0, 2) * 127).astype(np.uint8).transpose((0, 2, 3, 1))
            images.append(surfnorm)
        elif key == "semantic":
            titles.append(key)
            normalize.append(False)
            seg = (val * 314.159 % 255).astype(np.uint8)
            seg = np.reshape(seg, (-1, seg.shape[-1]))
            images.append(seg)
        elif key == "output_reconstruction":
            titles.append(key)
            normalize.append(False)
            val = np.clip(val, -0.5, 0.5)
            val += 0.5
            val *= 255
            val = val.astype(np.uint8).transpose((0, 2, 3, 1))
            images.append(val)
        elif key in {"action_prob", "action_taken", "egomotion_pred", "best_next_action"}:
            if key == "action_prob":
                titles.append(("Output Distribution", "p(Forward)     p(Left)     p(Right)"))
            else:
                titles.append(key)
            if val is not None:
                normalize.append(True)
                prob_hists = np.concatenate([draw_probability_hist(pi) for pi in val.copy()], axis=0)
                images.append(prob_hists)

            else:
                images.append(None)
                normalize.append(False)
    images.append(top_down_map)
    normalize.append(True)
    titles = [string.capwords(title.replace("_", " ")) if isinstance(title, str) else title for title in titles]
    return images, titles, normalize
