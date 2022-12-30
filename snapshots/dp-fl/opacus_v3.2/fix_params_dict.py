import torch

PATH = "/home/aheyler/PAC-pred-set/snapshots/opacus_v3.2/model_params_best"
SAVING = True

state_dict = torch.load(PATH)
bad_keys = list(state_dict.keys())
print(len(bad_keys))


keys = ['model.conv1.weight', 'model.bn1.weight', 'model.bn1.bias', 'model.bn1.running_mean', 'model.bn1.running_var', 'model.bn1.num_batches_tracked', 'model.layer1.0.conv1.weight', 'model.layer1.0.bn1.weight', 'model.layer1.0.bn1.bias', 'model.layer1.0.bn1.running_mean', 'model.layer1.0.bn1.running_var', 'model.layer1.0.bn1.num_batches_tracked', 'model.layer1.0.conv2.weight', 'model.layer1.0.bn2.weight', 'model.layer1.0.bn2.bias', 'model.layer1.0.bn2.running_mean', 'model.layer1.0.bn2.running_var', 'model.layer1.0.bn2.num_batches_tracked', 'model.layer1.1.conv1.weight', 'model.layer1.1.bn1.weight', 'model.layer1.1.bn1.bias', 'model.layer1.1.bn1.running_mean', 'model.layer1.1.bn1.running_var', 'model.layer1.1.bn1.num_batches_tracked', 'model.layer1.1.conv2.weight', 'model.layer1.1.bn2.weight', 'model.layer1.1.bn2.bias', 'model.layer1.1.bn2.running_mean', 'model.layer1.1.bn2.running_var', 'model.layer1.1.bn2.num_batches_tracked', 'model.layer2.0.conv1.weight', 'model.layer2.0.bn1.weight', 'model.layer2.0.bn1.bias', 'model.layer2.0.bn1.running_mean', 'model.layer2.0.bn1.running_var', 'model.layer2.0.bn1.num_batches_tracked', 'model.layer2.0.conv2.weight', 'model.layer2.0.bn2.weight', 'model.layer2.0.bn2.bias', 'model.layer2.0.bn2.running_mean', 'model.layer2.0.bn2.running_var', 'model.layer2.0.bn2.num_batches_tracked', 'model.layer2.0.downsample.0.weight', 'model.layer2.0.downsample.1.weight', 'model.layer2.0.downsample.1.bias', 'model.layer2.0.downsample.1.running_mean', 'model.layer2.0.downsample.1.running_var', 'model.layer2.0.downsample.1.num_batches_tracked', 'model.layer2.1.conv1.weight', 'model.layer2.1.bn1.weight', 'model.layer2.1.bn1.bias', 'model.layer2.1.bn1.running_mean', 'model.layer2.1.bn1.running_var', 'model.layer2.1.bn1.num_batches_tracked', 'model.layer2.1.conv2.weight', 'model.layer2.1.bn2.weight', 'model.layer2.1.bn2.bias', 'model.layer2.1.bn2.running_mean', 'model.layer2.1.bn2.running_var', 'model.layer2.1.bn2.num_batches_tracked', 'model.layer3.0.conv1.weight', 'model.layer3.0.bn1.weight', 'model.layer3.0.bn1.bias', 'model.layer3.0.bn1.running_mean', 'model.layer3.0.bn1.running_var', 'model.layer3.0.bn1.num_batches_tracked', 'model.layer3.0.conv2.weight', 'model.layer3.0.bn2.weight', 'model.layer3.0.bn2.bias', 'model.layer3.0.bn2.running_mean', 'model.layer3.0.bn2.running_var', 'model.layer3.0.bn2.num_batches_tracked', 'model.layer3.0.downsample.0.weight', 'model.layer3.0.downsample.1.weight', 'model.layer3.0.downsample.1.bias', 'model.layer3.0.downsample.1.running_mean', 'model.layer3.0.downsample.1.running_var', 'model.layer3.0.downsample.1.num_batches_tracked', 'model.layer3.1.conv1.weight', 'model.layer3.1.bn1.weight', 'model.layer3.1.bn1.bias', 'model.layer3.1.bn1.running_mean', 'model.layer3.1.bn1.running_var', 'model.layer3.1.bn1.num_batches_tracked', 'model.layer3.1.conv2.weight', 'model.layer3.1.bn2.weight', 'model.layer3.1.bn2.bias', 'model.layer3.1.bn2.running_mean', 'model.layer3.1.bn2.running_var', 'model.layer3.1.bn2.num_batches_tracked', 'model.layer4.0.conv1.weight', 'model.layer4.0.bn1.weight', 'model.layer4.0.bn1.bias', 'model.layer4.0.bn1.running_mean', 'model.layer4.0.bn1.running_var', 'model.layer4.0.bn1.num_batches_tracked', 'model.layer4.0.conv2.weight', 'model.layer4.0.bn2.weight', 'model.layer4.0.bn2.bias', 'model.layer4.0.bn2.running_mean', 'model.layer4.0.bn2.running_var', 'model.layer4.0.bn2.num_batches_tracked', 'model.layer4.0.downsample.0.weight', 'model.layer4.0.downsample.1.weight', 'model.layer4.0.downsample.1.bias', 'model.layer4.0.downsample.1.running_mean', 'model.layer4.0.downsample.1.running_var', 'model.layer4.0.downsample.1.num_batches_tracked', 'model.layer4.1.conv1.weight', 'model.layer4.1.bn1.weight', 'model.layer4.1.bn1.bias', 'model.layer4.1.bn1.running_mean', 'model.layer4.1.bn1.running_var', 'model.layer4.1.bn1.num_batches_tracked', 'model.layer4.1.conv2.weight', 'model.layer4.1.bn2.weight', 'model.layer4.1.bn2.bias', 'model.layer4.1.bn2.running_mean', 'model.layer4.1.bn2.running_var', 'model.layer4.1.bn2.num_batches_tracked', 'model.fc.weight', 'model.fc.bias']
good_keys = []
for k in keys: 
    if 'running' not in k and 'num_batches_tracked' not in k: 
        good_keys.append(k)
print(len(good_keys))

mapping = {bad_keys[i]: good_keys[i] for i in range(62)}

new_dict = {}
for k in bad_keys: 
    good_k = mapping[k]
    new_dict[good_k] = state_dict[k]

print(new_dict.keys())

if SAVING: 
    torch.save(new_dict, PATH)
    torch.save(state_dict, PATH + " copy")