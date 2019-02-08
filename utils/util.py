import json
from easydict import EasyDict
import os
import cPickle as cp
from os import listdir


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'rb') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config, config_dict

def choose_group(data_dir):
    config, _ = get_config_from_json("../configs/config_RL.json")
    filename = data_dir + 'target_group.txt'
    if os.path.exists(filename):
        ff = open(filename,'rb')
        return ff.readlines()[0]
    else:
        f = sorted(os.listdir(data_dir))[0]
        with open(data_dir + f, 'rb') as infile:
            reqlist = cp.load(infile)
        d = {}
        length = len(reqlist)
        for i,req in enumerate(reqlist):
            print "count and choose:",i, length
            if req[1][:config.PREFIX] not in d:
                d[req[1][:config.PREFIX]] = 1
            else:
                d[req[1][:config.PREFIX]] += 1

        target_group = sorted(d.items(), key=lambda x: x[1])[-1][0]
        print target_group
        ff = open(filename,'wb')
        ff.write(target_group)
        # choose group with most reqs in first day
        return target_group

def index_contents(dir):
    in_dir = '../data/req/'
    out_dir = '../data/req_index/'
    index = 0
    d = {}

    for f in sorted(listdir(in_dir)):
        if not f.endswith('cp'):
            continue
        with open(in_dir + f, 'rb') as infile:
            reqlist = cp.load(infile)

        req_list_indexed = []
        cnt = 0
        total = len(reqlist)
        for req in reqlist:
            cnt += 1
            print('indexing contents : ' + str(cnt) + '/'+str(total))
            if req[2] not in d:
                d[req[2]] = index
                index += 1
            req_list_indexed.append([req[0], req[1], d[req[2]]])

        with open(out_dir + f, 'wb') as outfile:
            cp.dump(req_list_indexed, outfile)
    with open(out_dir + 'title_map.cp', 'wb') as outfile:
        cp.dump(d, outfile)


