import numpy as np


base_class = dict(RNFL=1, GCL=2, IPL=3, INL=4, OPL=5, ONL=6, IS=7, OS=8, RPE=9)
dataset_dict = dict(HC_MS=dict(RNFL=1, GCL_IPL=2, INL=3, OPL=4, ONL=5, IS=6, OS=7, RPE=8),
                    DME=dict(RNFL=1, GCL_IPL=2, INL=3, OPL=4, ONL=5, IS=6, OS_RPE=7),
                    GCN=dict(RNFL=1, GCL=2, IPL=3, INL=4, OPL=5, ONL=6, IS_OS=7, RPE=8),
                    A2A_SDOCT=dict(RNFL_OS=1, RPE=2),
                    AMD=dict(RNFL_OS=1, RPE=2),
                    #Goals=dict(RNFL=1, GCL_IPL=2, INL_RPE=3),
                    HEG=dict(RNFL=1, GCL_IPL=2, INL=3, OPL=4, ONL_IS=5, OS=6, RPE=7),
                    OCTA_500=dict(RNFL_IPL=1, INL_OPL=2, ONL_IS=3, OS=4, RPE=5),
                    NR206=dict(RNFL=1, GCL_IPL=2, INL=3, OPL=4, ONL=5, IS=6, OS=7, RPE=8))


def generate_all_taskonehot():
    taskonehots = []
    layer_names = list(base_class.keys())
    names = []
    for i in range(9):
        taskonehot = np.zeros(9)
        name_axis=[]
        for key in range(i, len(layer_names)):
            name_axis.append(base_class[layer_names[key]] - 1)
            taskonehot[name_axis] = 1
            taskonehots.append(np.copy(taskonehot))
            if i != key:
                names.append(layer_names[i]+'_'+layer_names[key])
            else:
                names.append(layer_names[i])
    # taskonehots = np.array(taskonehots)
    labels = np.zeros([len(taskonehots), 512, 512])
    return names, taskonehots, labels


def new_layers_generate_v2(onehot_layers, layer_names, task_onehot):
    new_task_onehots = []
    new_layers = []
    new_layer_names = []
    times = len(onehot_layers)
    gap = 0
    for t in range(times):
        for i in range(len(task_onehot)):
            new_layer = np.zeros(onehot_layers[0].shape)
            onehot = np.zeros(task_onehot[0].shape[0])
            for j in range(i+1+gap, len(task_onehot)):
                for layer in range(i, j+1):
                    new_layer[onehot_layers[layer]!=0]=1
                    onehot += task_onehot[layer]

                new_task_onehots.append(onehot)
                new_layers.append(new_layer)
                start = layer_names[i].split('_')[0]
                if '_' in layer_names[j]:
                    end = layer_names[j].split('_')[1]
                else:
                    end = layer_names[j]
                new_layer_names.append(start+'_'+end)
                break
        gap+=1
    return new_layers, new_layer_names, new_task_onehots

def layer2onehot_label(dataset_name, layer):
    onehot_layers = [(layer == i).astype(np.int8) for i in np.unique(layer) if i != 0]

    layer_names = []
    task_onehot = []
    for k in dataset_dict[dataset_name].keys():
        for i in np.unique(layer):
            if dataset_dict[dataset_name][k] == i:
                layer_names.append(k)
                # boundary.setdefault(k, dict(top=np.argmax(), bottom=0))
                task_onehot.append(np.array(layer_name2onehot(k)))
    # if len(onehot_layers) != len(layer_names):
    #     print(0)
    return onehot_layers, layer_names, task_onehot

def layer_name2onehot(name):
    onehot = np.zeros(9)
    name_axis = []
    for key in base_class.keys():
        if key in name:
            name_axis.append(base_class[key] - 1)

    if len(name_axis) == 2:
        onehot[name_axis[0]: name_axis[1] + 1] = 1
    else:
        onehot[name_axis[0]] = 1
    return onehot

def onehot2layername(onehot):
    x = np.array(np.where(onehot == 1)) + 1
    for key in base_class.keys():
        if x.shape[1] >= 2:
            if base_class[key] == x[0][0]:
                    name1 = key
            if base_class[key] == x[0][-1]:
                name2 = key
                layer_name = name1 + '_' + name2
                return layer_name
        else:
            if base_class[key] == x[0][0]:
                layer_name = key
                return layer_name