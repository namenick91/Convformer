import os
import json

results = {}
metrics = ['mae', 'mse', 'rmse', 'mape', 'mspe']

directory = os.fsencode('../generated')
for run in os.listdir(directory):
    name = str(run)[:-(len(str(run).split('_')[-1])+1)]
    if name not in results:
        results[name] = {}
        results[name]['metrics'] = {}
        for metric in metrics:
            results[name]['metrics'][metric] = 0

        results[name]['time'] = {}
        results[name]['time']['train'] = 0
        results[name]['time']['test']  = 0

        results[name]['runs'] = 1
    else:
        results[name]['runs'] += 1

    run_dir = os.path.join(directory, run)
    for file in os.listdir(run_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".jsonl"):
            results_path = os.path.join(run_dir, file)
            with open(results_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line.strip())
                    if 'metrics' in line:
                        for k, v in line['metrics'].items():
                            results[name]['metrics'][k] += v
                    if 'time' in line:
                        results[name]['time']['train'] += line['time']['train']['seconds']
                        results[name]['time']['test']  += line['time']['test']['seconds']

with open('../results.txt', 'w') as f:
    for run in results:
        count = results[run]['runs']
        
        average_metric_values = {}
        for metric in metrics:
            average_metric_values[metric] = results[run]['metrics'][metric] / count
        
        average_train_time = results[run]['time']['train'] / count
        average_test_time  = results[run]['time']['test']  / count

        f.write(f'{run[2:]}({count})\n')
        for k,v in average_metric_values.items():
            f.write(f'{k}: {v}\n')
        for k, v in zip(['train', 'test'], [average_train_time, average_test_time]):
            secs = v
            minutes = int(secs // 60)
            rem = secs - 60 * minutes

            f.write(f'{k}: {secs}s ({minutes}m {rem}s)\n')
        f.write('\n')
