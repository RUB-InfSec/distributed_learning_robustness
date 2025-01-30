import os
import ujson as json

from utils.eval_utils import get_raw_data, get_results_from_data

from utils.nutils import *

batch_size = 500

def eval_eff(model, loader, device="cpu"):
    # Evaluate the model on the benign and attack data
    model.eval()
    model.to(device)

    # store the model output for acc and robust acc
    acc_out = []
    labels = []

    with torch.no_grad():
        for _, pck in enumerate(loader):
            data, target = pck
            data, target = data.to(device), target.to(device)
            output = model(data)

            acc_out.append(output.cpu().numpy().tolist())
            labels.append(target.cpu().numpy().tolist())
    return acc_out, labels


def load_adv_dataset(folder, transform, batch_size=batch_size):
    files = {}
    for file in os.listdir(folder):
        if "labels" in file:
            labels = np.load(os.path.join(folder, file))
        else:
            batchname = file.split("_")[1].split(".")[0]
            files[batchname] = os.path.join(folder, file)
    images = []
    for i in range(len(files.keys())):
        # test if all the files are there
        if str(i) not in files.keys():
            print("Error: Missing file")
            return

        batch = np.load(files[str(i)])
        topil = transforms.ToPILImage()
        for image in batch:
            images.append(topil(np.transpose(image, (1, 2, 0))))

    dataset = SelfDataset(images, labels, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=10)


def evaluate_models(eps, device='cuda:0'):
    benign_set = get_dataset("cifar10", train=False, path="data/cifar10")

    transform = transforms.Compose(
        [transforms.ToImage(),
         transforms.ToDtype(torch.float32, scale=True),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    benign_loader = torch.utils.data.DataLoader(benign_set, batch_size=batch_size, shuffle=False, num_workers=4)

    mi_sa_mloader = load_adv_dataset(f"{attack_datasets_path}/MI_SAM-eps_{eps}", transform=transform)
    mi_cw_loader = load_adv_dataset(f"{attack_datasets_path}/MI_CommonWeakness-eps_{eps}",
                                   transform=transform)
    mi_cse_loader = load_adv_dataset(
        f"{attack_datasets_path}/MI_CosineSimilarityEncourager-eps_{eps}", transform=transform)
    loaders = [benign_loader, mi_sa_mloader, mi_cw_loader, mi_cse_loader]
    names = ["benign", "MI_SAM", "MI_CW", "MI_CSE"]

    versions = list(filter(lambda a: "log" in a, os.listdir(ray_log_path)))

    for v in versions:
        vname = v.replace(".json", "").replace(".log", "")
        fname = f"{processed_data_path}/results_{vname}_eps_{eps}.json"
        if os.path.isfile(fname):
            print(
                f"Skipping {v} as it is already processed. Empty the `processed_data` folder if you want to reprocess it.")
            continue

        jsf = False
        file_path = os.path.join(ray_log_path, v)
        if ".log" in v:
            lines = []
            if os.path.isfile(file_path):
                with open(file_path, 'r') as fi:
                    lines.extend(fi.readlines())
            jsonLines = []
            i = 0
            while i < len(lines):
                l = lines[i]
                if l[0] == "{":
                    jsonLines.append(l)
                i += 1

            paramJson = [json.loads(l.replace("\n", "").replace("'", "\"").replace("None", "\"None\"")) for l in
                         jsonLines]
        elif "json" in v:
            jsf = True
            if os.path.isfile(file_path):
                with open(file_path, 'r') as fi:
                    paramJson = json.load(fi)
        else:
            raise NotImplementedError(f"unknown file type {v}")

        results = {}
        for pj in paramJson:
            if jsf:
                pj = paramJson[pj]

            if "status" in pj and pj["status"] == "ERROR":
                continue

            if "converging_threshold" in pj and (pj['converging_threshold'] != "None") and pj["test_acc"] < pj[
                'converging_threshold']:
                continue

            model = get_model("cifar10", pj["config"]["arch"])
            path = pj["path"]
            mpath = os.path.join(path, "model.pt")
            model_state_dict = torch.load(mpath)
            model.load_state_dict(model_state_dict)

            for loader, lname in list(zip(loaders, names)):
                n = pj["config"]["node_id"]
                N = pj["config"]["node_count"]
                r = pj["round"]
                split = pj['config']["datasplit"]

                if split not in results:
                    results[split] = {}
                if r not in results[split]:
                    results[split][r] = {}
                if N not in results[split][r]:
                    results[split][r][N] = {}
                if lname not in results[split][r][N]:
                    results[split][r][N][lname] = {}
                acc_out, labels = eval_eff(model, loader, device=device)
                results[split][r][N][lname][n] = {"acc_out": acc_out, "labels": labels, 'config': pj}

        with open(fname, "w") as file:
            json.dump(results, file)


if __name__ == "__main__":
    ray_log_path = "data/ray_logs"
    processed_data_path = "data/processed_data"
    attack_datasets_path = "data/attacks"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_models(eps=8, device=device)

    raw_data = get_raw_data(processed_data_path)
    results = get_results_from_data(raw_data)

    print(results)