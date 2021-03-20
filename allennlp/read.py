import pickle
with open('phrase1.pk', 'rb') as f:
    data = pickle.load(f)

with open('corrected_1.txt', 'a') as w:
    for k,v in data.items():
        print('[info]key is:{}'.format(k))
        # pickle.dump(k, w)
        w.write('[info]key is:{} \n'.format(k))
        v_set = set(v)
        for x in v_set:
            print('{}'.format(x))
            # pickle.dump(x, w)
            w.write(x + '\n')