import sys
import json

config_file = sys.argv[1]
directory = sys.argv[2]

try:
    config = json.load(open(config_file, 'r'))
    scripts = [{'name':[],'script':[]}] 

    for stage in config:
        new_scripts = []
        for c in stage:
            for s in scripts:
                ns = {
                        'name': s['name'] + ([c['name']] if c['name'] != "" else []),
                        'script': s['script'] + [c['params']]
                    }
                new_scripts.append(ns)
        scripts = new_scripts
                
    for s in scripts:
        name = "_".join(s['name'])
        script = " ".join(s['script'])
        try:
            f = open('{}/{}.sh'.format(directory, name), 'w')
            f.write(script)
        except:
            print('Cant open file {}/{}.sh'.format(directory, s['name']))

        print("_".join(s['name']))
        print(" ".join(s['script']))

except:
    print('Cant open file with configs: {}'.format(config_file))
    raise

