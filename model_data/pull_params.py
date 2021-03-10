import sys
import os
import pandas as pd
from os import listdir
import pickle

'''
Reading the std-error files to pull out the parameters for files that do not have rules in them

path - location to the *.out and *.err files
files - list of all the .err files
dict_name - name to call the pickled dictionary
'''
def no_rules(path,files,dict_name):

    dict = {}
    for f in files:

        nf = f.split('.')[0]
        out_f = nf + '.out'

        out_r = pd.read_csv(path + out_f, sep='\t', names=['model', 'a', 'b', 'test', 'd'])

        params = ''

        # finding the specific lines we want to extract
        of = open(path + f, 'r+')
        for line in of:
            if 'Best parameters:' in line:
                params = line.split('parameters:')[1].lstrip().rstrip()
                line = of.readline()
                while 'feature importances' not in line and '[' not in line and 'Feature ranking' not in line and 'Top 5 best Parameters:' not in line and 'accuracy' not in line:
                    params += line.lstrip().rstrip()
                    line = of.readline()
                break

        try:
            # make the dictionary we want to store
            if '_dm_' in out_r.a.loc[0] or '_dm.' in out_r.a.loc[0] or ( '_dm_' not in out_r.a.loc[0] and '_dm.' not in out_r.a.loc[0]):

                if out_r.model.loc[0] not in dict and (
                        out_r.d.loc[0].lstrip().rstrip() == 'minimal' or out_r.d.loc[0].lstrip().rstrip() == 'subset'):

                    dict[out_r.model.loc[0]] = {
                        'minimal': { 'BBE': {'score': 0.0, 'param': {}},'og': {'score': 0.0, 'param': {}}},
                        'subset': {'BBE': {'score': 0.0, 'param': {}}, 'og': {'score': 0.0, 'param': {}}}}

                    if out_r.d.loc[0] == 'minimal':
                        if '_dm_' in out_r.a.loc[0] or '_dm.' in out_r.a.loc[0]:
                            dict[out_r.model.loc[0]]['minimal']['BBE'] = {'score': out_r.test.loc[0],
                                                                              'param': params}
                        elif '_dm_' not in out_r.a.loc[0] and '_dm.' not in out_r.a.loc[0]:
                            dict[out_r.model.loc[0]]['minimal']['og'] = {'score': out_r.test.loc[0],
                                                                             'param': params}

                    elif out_r.d.loc[0] == 'subset':
                        if '_dm_' in out_r.a.loc[0] or '_dm.' in out_r.a.loc[0]:
                            dict[out_r.model.loc[0]]['subset']['BBE'] = {'score': out_r.test.loc[0],
                                                                             'param': params}
                        elif '_dm_' not in out_r.a.loc[0] and '_dm.' not in out_r.a.loc[0]:
                            dict[out_r.model.loc[0]]['subset']['og'] = {'score': out_r.test.loc[0], 'param': params}

                else:
                    if out_r.d.loc[0] == 'minimal':

                        if '_dm_' in out_r.a.loc[0] or '_dm.' in out_r.a.loc[0]:
                            if dict[out_r.model.loc[0]]['minimal']['BBE']['score'] < out_r.test.loc[0]:
                                dict[out_r.model.loc[0]]['minimal']['BBE'] = {'score': out_r.test.loc[0],
                                                                                  'param': params}
                        elif '_dm_' not in out_r.a.loc[0] and '_dm.' not in out_r.a.loc[0]:
                            if dict[out_r.model.loc[0]]['minimal']['og']['score'] < out_r.test.loc[0]:
                                dict[out_r.model.loc[0]]['minimal']['og'] = {'score': out_r.test.loc[0],
                                                                                 'param': params}

                    elif out_r.d.loc[0] == 'subset':

                        if '_dm_' in out_r.a.loc[0] or '_dm.' in out_r.a.loc[0]:
                            if dict[out_r.model.loc[0]]['subset']['BBE']['score'] < out_r.test.loc[0]:
                                dict[out_r.model.loc[0]]['subset']['BBE'] = {'score': out_r.test.loc[0],
                                                                                 'param': params}
                        elif '_dm_' not in out_r.a.loc[0] and '_dm.' not in out_r.a.loc[0]:
                            if dict[out_r.model.loc[0]]['subset']['og']['score'] < out_r.test.loc[0]:
                                dict[out_r.model.loc[0]]['subset']['og'] = {'score': out_r.test.loc[0],
                                                                                'param': params}

        except:
            continue

    out_dict = {}
    for model in dict:
        for fs in dict[model]:
            for addon in dict[model][fs]:
                out_dict[(model, fs, addon)] = dict[model][fs][addon]['param']

    with open(dict_name, 'wb') as handle:
        pickle.dump(out_dict, handle)

'''
Reading the std-error files to pull out the parameters for files that have rules in them

path - location to the *.out and *.err files
files - list of all the .err files
dict_name - name to call the pickled dictionary
'''
def rules(path,files,dict_name):

    dict = {}
    for f in files:

        nf = f.split('.')[0]
        out_f = nf + '.out'

        out_r = pd.read_csv(path + out_f, sep='\t', names=['model', 'a', 'b', 'test', 'd'])

        params = ''
        of = open(path + f, 'r+')
        for line in of:
            if 'Best parameters:' in line:
                params = line.split('parameters:')[1].lstrip().rstrip()
                line = of.readline()
                while 'feature importances' not in line and '[' not in line and 'Feature ranking' not in line and 'Top 5 best Parameters:' not in line and 'accuracy' not in line:
                    params += line.lstrip().rstrip()
                    line = of.readline()
                break

        try:

            if '_dm_' in out_r.a.loc[0] or '_dm.' in out_r.a.loc[0] or ( '_dm_' not in out_r.a.loc[0] and '_dm.' not in out_r.a.loc[0]):

                lowered = out_r.a.loc[0].lower()
                x = 'simp' if 'simp' in lowered else 'foil'

                if out_r.model.loc[0] not in dict and (
                        out_r.d.loc[0].lstrip().rstrip() == 'minimal' or out_r.d.loc[0].lstrip().rstrip() == 'subset'):

                    dict[out_r.model.loc[0]] = {
                        'minimal': {'simp': {'BBE': {'score': 0.0, 'param': {}},
                                             'og': {'score': 0.0, 'param': {}}},
                                    'foil': {'BBE': {'score': 0.0, 'param': {}},
                                             'og': {'score': 0.0, 'param': {}}}},
                        'subset': {'simp': {'BBE': {'score': 0.0, 'param': {}},
                                            'og': {'score': 0.0, 'param': {}}},
                                   'foil': {'BBE': {'score': 0.0, 'param': {}},
                                            'og': {'score': 0.0, 'param': {}}}}}

                    if out_r.d.loc[0] == 'minimal':

                        if '_dm_' in out_r.a.loc[0] or '_dm.' in out_r.a.loc[0]:
                            dict[out_r.model.loc[0]]['minimal'][x]['BBE'] = {'score': out_r.test.loc[0],
                                                                                   'param': params}
                        elif '_dm_' not in out_r.a.loc[0] and '_dm.' not in out_r.a.loc[0]:
                            dict[out_r.model.loc[0]]['minimal'][x]['og'] = {'score': out_r.test.loc[0],
                                                                                  'param': params}

                    elif out_r.d.loc[0] == 'subset':

                        if '_dm_' in out_r.a.loc[0] or '_dm.' in out_r.a.loc[0]:
                            dict[out_r.model.loc[0]]['subset'][x]['BBE'] = {'score': out_r.test.loc[0],
                                                                                  'param': params}
                        elif '_dm_' not in out_r.a.loc[0] and '_dm.' not in out_r.a.loc[0]:
                            dict[out_r.model.loc[0]]['subset'][x]['og'] = {'score': out_r.test.loc[0],
                                                                                 'param': params}

                else:
                    if out_r.d.loc[0] == 'minimal':

                        if '_dm_' in out_r.a.loc[0] or '_dm.' in out_r.a.loc[0]:
                            if dict[out_r.model.loc[0]]['minimal'][x]['BBE']['score'] < out_r.test.loc[0]:
                                dict[out_r.model.loc[0]]['minimal'][x]['BBE'] = {'score': out_r.test.loc[0],
                                                                                       'param': params}
                        elif '_dm_' not in out_r.a.loc[0] and '_dm.' not in out_r.a.loc[0]:
                            if dict[out_r.model.loc[0]]['minimal'][x]['og']['score'] < out_r.test.loc[0]:
                                dict[out_r.model.loc[0]]['minimal'][x]['og'] = {'score': out_r.test.loc[0],
                                                                                      'param': params}

                    if out_r.d.loc[0] == 'subset':

                        if '_dm_' in out_r.a.loc[0] or '_dm.' in out_r.a.loc[0]:
                            if dict[out_r.model.loc[0]]['subset'][x]['BBE']['score'] < out_r.test.loc[0]:
                                dict[out_r.model.loc[0]]['subset'][x]['BBE'] = {'score': out_r.test.loc[0],
                                                                                      'param': params}
                        elif '_dm_' not in out_r.a.loc[0] and '_dm.' not in out_r.a.loc[0]:
                            if dict[out_r.model.loc[0]]['subset'][x]['og']['score'] < out_r.test.loc[0]:
                                dict[out_r.model.loc[0]]['subset'][x]['og'] = {'score': out_r.test.loc[0],
                                                                                     'param': params}

        except:
            continue

    out_dict = {}
    for model in dict:
        for fs in dict[model]:
            for rule in dict[model][fs]:
                for addon in dict[model][fs][rule]:
                    out_dict[(model, fs, rule, addon)] = dict[model][fs][rule][addon]['param']

    with open(dict_name, 'wb') as handle:
        pickle.dump(out_dict, handle)

if __name__ == '__main__':

    # path = 'models/law2vec/law2vec/'
    path = sys.argv[1]
    # p_rules = '-r'
    p_rules = sys.argv[2]
    # dict_name = 'models/l2v_r_dict.pickle'
    dict_name = sys.argv[3]

    files = [x for x in listdir(path) if '.err' in x]

    if p_rules == '-r':
        rules(path, files, dict_name)
    else:
        no_rules(path,files,dict_name)

