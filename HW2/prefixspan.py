import os
import csv


def prefixspan(database, minsup, prefix, result):
    if len(database) < 1:
        return
    prefix = prefix
    db = database
    count = {}
    project = {}
    for tran in db:
        visited = set()
        for i, item in enumerate(tran):
            if item not in count:
                count[item] = 1
                project[item] = []
                _project_item = [s for s in tran[i+1:]]
                if len(_project_item) > 0:
                    project[item].append(_project_item)
                visited.add(item)
            elif (item in count) and (item not in visited):
                count[item] += 1
                _project_item = [s for s in tran[i+1:]]
                if len(_project_item) > 0:
                    project[item].append(_project_item)
                visited.add(item)

    remove = []
    for item, ncount in count.items():
        if ncount < minsup:
            remove.append(item)
        else:
            r = prefix + [item]
            result.append((r, ncount))
    for item in remove:
        del count[item]
        del project[item]

    for item, project_db in project.items():
        next_prefix = prefix + [item]
        prefixspan(project_db, minsup, next_prefix, result)
    return


def dump_result(result, output_dir):
    result_sorted = sorted(result)
    with open(output_dir, 'w') as f:
        for sequence, count in result_sorted:
            f.write('[')
            for i, word in enumerate(sequence):
                f.write(str(word))
                if i != len(sequence) - 1:
                    f.write(',')
            f.write('],{}\n'.format(count))


def read_database(input_dir):
    db = []
    with open(input_dir, newline='') as csvfile:
        rows = csv.reader(csvfile)

        for row in rows:
            clean_row = [int(s) for s in row]
            db.append(clean_row)
    return db


if __name__ == '__main__':
    # change working dir
    os.chdir('/home/jameschiu/workplace/CCBDA-2019-Spring-0750730/HW2')

    db = read_database('publicdataset_2.csv')

    result = []
    minsup = 2
    prefix = []
    prefixspan(db, minsup, prefix, result)

    dump_result(result, 'output_2.txt')
