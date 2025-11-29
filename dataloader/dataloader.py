from scipy.io import loadmat
import os
import torch

#处理测试集数据
def dataloader(f_path, f_name, tag):
    with torch.no_grad():
        # 获取当前脚本所在的目录
        current_path = os.path.dirname(os.path.abspath(__file__))
        # 构建文件路径
        file_path = os.path.join(current_path, "dataofEEG", f_path, f_name)

        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping...")
            return torch.tensor(-1),torch.tensor(-1)

        # 加载.mat文件
        mat_data = loadmat(file_path)

        # 提取数据矩阵
        data_key = [key for key in mat_data.keys() if not key.startswith("__")][0]
        eeg_matrix = mat_data[data_key]
        # 检查矩阵形状是否符合预期
        if eeg_matrix.shape != (43, 2, 5):
            print(f"Warning: File {f_name} has unexpected shape {eeg_matrix.shape}")

        # 添加到数据集
        X = torch.tensor(eeg_matrix, dtype=torch.float32).unsqueeze(0)
        Y = torch.tensor(tag).unsqueeze(0)

    return X,Y

def dataloader_de(name, start, end):
    with torch.no_grad():
        for i in range(start, end):
            name1 = f"{name}Normal"
            name2 = f"{name}Patient"
            f_name = f"._{i}.mat_de.mat"
            if (i == start):
                X, Y = dataloader(name1, f_name, 0)
                temp_X, temp_Y = dataloader(name2, f_name, 1)
                if not torch.equal(temp_X, torch.tensor(-1)):
                    X = torch.cat((X, temp_X), dim=0)
                    Y = torch.cat((Y, temp_Y), dim=0)
            else:
                temp_X, temp_Y = dataloader(name1, f_name, 0)
                if not torch.equal(temp_X, torch.tensor(-1)):
                    X = torch.cat((X, temp_X), dim=0)
                    Y = torch.cat((Y, temp_Y), dim=0)
                temp_X, temp_Y = dataloader(name2, f_name, 1)
                if not torch.equal(temp_X, torch.tensor(-1)):
                    X = torch.cat((X, temp_X), dim=0)
                    Y = torch.cat((Y, temp_Y), dim=0)

    return X,Y

def dataloader_psd(name, start, end):
    with torch.no_grad():
        for i in range(start, end):
            name1 = f"{name}Normal"
            name2 = f"{name}Patient"
            f_name = f"._{i}.mat_psd.mat"
            if (i == start):
                X, Y = dataloader(name1, f_name, 0)
                temp_X, temp_Y = dataloader(name2, f_name, 1)
                if not torch.equal(temp_X, torch.tensor(-1)):
                    X = torch.cat((X, temp_X), dim=0)
                    Y = torch.cat((Y, temp_Y), dim=0)
            else:
                temp_X, temp_Y = dataloader(name1, f_name, 0)
                if not torch.equal(temp_X, torch.tensor(-1)):
                    X = torch.cat((X, temp_X), dim=0)
                    Y = torch.cat((Y, temp_Y), dim=0)
                temp_X, temp_Y = dataloader(name2, f_name, 1)
                if not torch.equal(temp_X, torch.tensor(-1)):
                    X = torch.cat((X, temp_X), dim=0)
                    Y = torch.cat((Y, temp_Y), dim=0)

    return X,Y
