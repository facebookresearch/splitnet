import os
import random

import PIL
import cv2
import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.multiprocessing import set_start_method
from torchvision import transforms

from arguments import get_args
from dataset_dump.create_rgb_dataset import RandomImageGenerator
from networks import optimizers
from networks.networks import ShallowVisualEncoder
from utils import drawing
from utils import python_util
from utils import pytorch_util as pt_util
from utils import tensorboard_logger

args = get_args()

TEST_BATCH_SIZE = 64
EPOCHS = 200
BATCH_SIZE = 32
NUM_CLASSES = 41

NUM_BATCHES_PER_EPOCH = 1000
NUM_TEST_BATCHES_PER_EPOCH = 50

DEBUG = args.debug
USE_SEMANTIC = False

if DEBUG:
    BATCH_SIZE = 4
    TEST_BATCH_SIZE = 4


def draw_outputs(output, labels, mode):
    output[:, 4:7] = output[:, 4:7] / output[:, 4:7].norm(dim=1, keepdim=True)
    output = pt_util.to_numpy_array(output)
    labels = {key: pt_util.to_numpy_array(val) for key, val in labels.items()}
    if USE_SEMANTIC:
        labels["semantic"][:, 0, 1] = 0
        labels["semantic"][:, 0, 0] = 40
    for bb in range(output.shape[0]):
        output_on = output[bb]
        labels_on = {key: val[bb] for key, val in labels.items()}
        output_semantic = None
        if USE_SEMANTIC:
            output_semantic = np.argmax(output_on[7:], axis=0)
            output_semantic[0, 0] = 40
            output_semantic[0, 1] = 0
        images = [
            labels_on["rgb"].transpose(1, 2, 0),
            255 - np.clip((labels_on["depth"] + 0.5).squeeze() * 255, 0, 255),
            (np.clip(labels_on["surface_normals"] + 1, 0, 2) * 127).astype(np.uint8).transpose(1, 2, 0),
            labels_on["semantic"].squeeze().astype(np.uint8) if USE_SEMANTIC else None,
            np.clip((output_on[:3] + 0.5) * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0),
            255 - np.clip((output_on[3] + 0.5).squeeze() * 255, 0, 255),
            (np.clip(output_on[4:7] + 1, 0, 2) * 127).astype(np.uint8).transpose(1, 2, 0),
            output_semantic.astype(np.uint8) if USE_SEMANTIC else None,
        ]
        titles = ["rgb", "depth", "normals", "semantic", "rgb_pred", "depth_pred", "normals_pred", "semantic_pred"]

        image = drawing.subplot(
            images, 2, 4, 256, 256, titles=titles, normalize=[False, False, False, True, False, False, False, True]
        )
        cv2.imshow("im_" + mode, image[:, :, ::-1])
        cv2.waitKey(0)


def train_model(model, device, train_loader, optimizer, total_num_steps, logger, net_output_info, checkpoint_dir):
    try:
        model.train()
        if args.tensorboard:
            logger.network_conv_summary(model, total_num_steps)
        data_iter = iter(train_loader)
        for batch_idx in tqdm.tqdm(range(NUM_BATCHES_PER_EPOCH)):
            data = next(data_iter)
            labels = {key: val.to(device) for key, val in data.items()}
            labels["surface_normals"] = pt_util.depth_to_surface_normals(labels["depth"])
            data = labels["rgb"].detach()
            _, output, class_pred = model.forward(data, True)
            loss, loss_val, visual_loss_dict = optimizers.get_visual_loss(output, labels, net_output_info)
            object_loss = 0
            if "class_label" in labels:
                object_loss = optimizers.get_object_existence_loss(class_pred, labels["class_label"])
                loss = loss + object_loss
                object_loss = object_loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if DEBUG:
                draw_outputs(output, labels, "train")
            total_num_steps += 1
            if not args.no_weight_update and batch_idx % args.log_interval == 0:
                if args.tensorboard:
                    log_dict = {"loss/visual/0_total": loss.item()}
                    if "class_label" in labels:
                        log_dict["loss/visual/object_loss"] = object_loss
                    for key, val in visual_loss_dict.items():
                        log_dict["loss/visual/" + key] = val
                    logger.dict_log(log_dict, step=total_num_steps)
            if args.tensorboard:
                if batch_idx % 100 == 0:
                    logger.network_variable_summary(model, total_num_steps)

        if args.save_checkpoints and not args.no_weight_update:
            pt_util.save(model, checkpoint_dir, num_to_keep=5, iteration=total_num_steps)
        return total_num_steps
    except Exception as e:
        import traceback

        traceback.print_exc()
        if args.save_checkpoints and not args.no_weight_update:
            pt_util.save(model, checkpoint_dir, num_to_keep=-1, iteration=total_num_steps)
        raise e


def evaluate_model(model, device, test_loader, total_num_steps, logger, net_output_info):
    model.eval()
    loss_val_total = 0
    object_loss_total = 0
    test_loss_dict = None
    n_its = 0
    with torch.no_grad():
        data_iter = iter(test_loader)
        for batch_idx in tqdm.tqdm(range(NUM_TEST_BATCHES_PER_EPOCH)):
            data = next(data_iter)
            n_its += 1
            labels = {key: val.to(device) for key, val in data.items()}
            labels["surface_normals"] = pt_util.depth_to_surface_normals(labels["depth"])
            data = labels["rgb"]
            _, output, class_pred = model.forward(data, True)
            loss, loss_val, visual_loss_dict = optimizers.get_visual_loss(output, labels, net_output_info)
            object_loss = 0
            if "class_label" in labels:
                object_loss = optimizers.get_object_existence_loss(class_pred, labels["class_label"])
                loss = loss + object_loss
                object_loss = object_loss.item()
            if DEBUG:
                draw_outputs(output, labels, "eval")
            if test_loss_dict is None:
                test_loss_dict = visual_loss_dict
            else:
                for key in test_loss_dict:
                    test_loss_dict[key] += visual_loss_dict[key]

            object_loss_total += object_loss
            loss_val_total += loss.item()

    loss_val_total /= n_its
    object_loss_total /= n_its
    for key in test_loss_dict:
        test_loss_dict[key] /= n_its

    if args.tensorboard:
        log_dict = {"loss/visual/0_total": loss_val_total}
        if "class_label" in labels:
            log_dict["loss/visual/object_loss"] = object_loss_total
        for key, val in visual_loss_dict.items():
            log_dict["loss/visual/" + key] = val
        logger.dict_log(log_dict, step=total_num_steps)


class HabitatImageGenerator(torch.utils.data.Dataset):
    def __init__(
        self,
        gpu_ids,
        dataset_name,
        dataset_split,
        dataset_path,
        images_before_reset=100,
        sensors=None,
        transform=None,
        depth_transform=None,
        semantic_transform=None,
    ):
        self.transform = transform
        self.depth_transform = depth_transform
        self.semantic_transform = semantic_transform
        self.gpu_ids = gpu_ids
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.dataset_path = dataset_path
        self.sensors = sensors
        self.images_before_reset = images_before_reset
        self.image_generator = None
        self.worker_id = 0

    def worker_init_fn(self, worker_id):
        self.worker_id = worker_id

    def __len__(self):
        return 2 ** 31

    def __getitem__(self, item):
        # Ignore the item and just generate an image
        if self.image_generator is None:
            self.image_generator = RandomImageGenerator(
                self.gpu_ids[self.worker_id % len(self.gpu_ids)],
                self.dataset_name,
                self.dataset_split,
                self.dataset_path,
                self.images_before_reset,
                self.sensors,
            )
        data = self.image_generator.get_sample()
        seed = np.random.randint(2 ** 32)
        transformed_data = {}
        image = data["rgb"]
        depth = data["depth"]
        if self.transform is not None:
            random.seed(seed)
            image = self.transform(data["rgb"])
        image = np.asarray(image)
        transformed_data["rgb"] = pt_util.from_numpy(image.transpose(2, 0, 1))
        if self.depth_transform is not None:
            random.seed(seed)
            depth = self.depth_transform(depth)
        depth = np.asarray(depth)
        depth[depth == 0] = 1
        depth -= 0.5
        depth = pt_util.from_numpy(depth)[np.newaxis, ...]
        transformed_data["depth"] = depth

        if data["class_semantic"] is not None:
            semantic = data["class_semantic"]
            if self.semantic_transform is not None:
                random.seed(seed)
                semantic = self.semantic_transform(semantic[:, :, np.newaxis])
            semantic = np.asarray(semantic)
            unique = np.unique(semantic)
            class_label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
            class_label_vec[unique] = 1
            class_label_vec = pt_util.from_numpy(class_label_vec)
            transformed_data["semantic"] = torch.LongTensor(semantic.squeeze())
            transformed_data["class_label"] = class_label_vec
        return transformed_data


def main():
    torch_devices = [int(gpu_id.strip()) for gpu_id in args.pytorch_gpu_ids.split(",")]
    render_gpus = [int(gpu_id.strip()) for gpu_id in args.render_gpu_ids.split(",")]
    device = "cuda:" + str(torch_devices[0])

    decoder_output_info = [("reconstruction", 3), ("depth", 1), ("surface_normals", 3)]
    if USE_SEMANTIC:
        decoder_output_info.append(("semantic", 41))

    model = ShallowVisualEncoder(decoder_output_info)
    model = pt_util.get_data_parallel(model, torch_devices)
    model = pt_util.DummyScope(model, ["base", "visual_encoder"])
    model.to(device)

    print("Model constructed")
    print(model)

    train_transforms = transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomCrop(224)]
    )

    train_transforms_depth = transforms.Compose(
        [PIL.Image.fromarray, transforms.RandomHorizontalFlip(), transforms.RandomCrop(224), np.array]
    )

    train_transforms_semantic = transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomCrop(224)]
    )

    sensors = ["RGB_SENSOR", "DEPTH_SENSOR"] + (["SEMANTIC_SENSOR"] if USE_SEMANTIC else [])
    if args.dataset == "suncg":
        data_train = HabitatImageGenerator(
            render_gpus,
            "suncg",
            args.data_subset,
            "data/dumps/suncg/{split}/dataset_one_ep_per_scene.json.gz",
            images_before_reset=1000,
            sensors=sensors,
            transform=train_transforms,
            depth_transform=train_transforms_depth,
            semantic_transform=train_transforms_semantic,
        )
        print("Num train images", len(data_train))

        data_test = HabitatImageGenerator(
            render_gpus,
            "suncg",
            "val",
            "data/dumps/suncg/{split}/dataset_one_ep_per_scene.json.gz",
            images_before_reset=1000,
            sensors=sensors,
        )
    elif args.dataset == "mp3d":
        data_train = HabitatImageGenerator(
            render_gpus,
            "mp3d",
            args.data_subset,
            "data/dumps/mp3d/{split}/dataset_one_ep_per_scene.json.gz",
            images_before_reset=1000,
            sensors=sensors,
            transform=train_transforms,
            depth_transform=train_transforms_depth,
            semantic_transform=train_transforms_semantic,
        )
        print("Num train images", len(data_train))

        data_test = HabitatImageGenerator(
            render_gpus,
            "mp3d",
            "val",
            "data/dumps/mp3d/{split}/dataset_one_ep_per_scene.json.gz",
            images_before_reset=1000,
            sensors=sensors,
        )
    elif args.dataset == "gibson":
        data_train = HabitatImageGenerator(
            render_gpus,
            "gibson",
            args.data_subset,
            "data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz",
            images_before_reset=1000,
            sensors=sensors,
            transform=train_transforms,
            depth_transform=train_transforms_depth,
            semantic_transform=train_transforms_semantic,
        )
        print("Num train images", len(data_train))

        data_test = HabitatImageGenerator(
            render_gpus,
            "gibson",
            "val",
            "data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz",
            images_before_reset=1000,
            sensors=sensors,
        )
    else:
        raise NotImplementedError("No rule for this dataset.")

    print("Num train images", len(data_train))
    print("Num val images", len(data_test))

    print("Using device", device)
    print("num cpus:", args.num_processes)

    train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=BATCH_SIZE,
        num_workers=args.num_processes,
        worker_init_fn=data_train.worker_init_fn,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=TEST_BATCH_SIZE,
        num_workers=len(render_gpus) if args.num_processes > 0 else 0,
        worker_init_fn=data_test.worker_init_fn,
        shuffle=False,
        pin_memory=True,
    )

    log_prefix = args.log_prefix
    time_str = python_util.get_time_str()
    checkpoint_dir = os.path.join(log_prefix, args.checkpoint_dirname, time_str)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_iter = 0
    if args.load_model:
        start_iter = pt_util.restore_from_folder(model, os.path.join(log_prefix, args.checkpoint_dirname, "*"))

    train_logger = None
    test_logger = None
    if args.tensorboard:
        train_logger = tensorboard_logger.Logger(
            os.path.join(log_prefix, args.tensorboard_dirname, time_str + "_train")
        )
        test_logger = tensorboard_logger.Logger(os.path.join(log_prefix, args.tensorboard_dirname, time_str + "_test"))

    total_num_steps = start_iter

    if args.save_checkpoints and not args.no_weight_update:
        pt_util.save(model, checkpoint_dir, num_to_keep=5, iteration=total_num_steps)

    evaluate_model(model, device, test_loader, total_num_steps, test_logger, decoder_output_info)

    for epoch in range(0, EPOCHS + 1):
        total_num_steps = train_model(
            model, device, train_loader, optimizer, total_num_steps, train_logger, decoder_output_info, checkpoint_dir
        )
        evaluate_model(model, device, test_loader, total_num_steps, test_logger, decoder_output_info)


if __name__ == "__main__":
    set_start_method("forkserver", force=True)
    main()
