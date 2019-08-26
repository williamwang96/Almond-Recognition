import matplotlib.pyplot as plt
import math

# Euler configuration
# 7 GPU: gtx titan x, each with ~11GB memory
# 8 CPU
# maximum resource available for request for a single job
# in the student partition

log_file_name = {0 : "logs-c1b1s13-30min.txt",
                 1 : "logs-c1b1s14-30min.txt",
                 2 : "logs-c1b1s16-30min.txt",
                 3 : "logs-c1b1s20-30min.txt",
                 4 : "logs-c1b1s25-30min.txt",
                 5 : "logs-c1b1s33-30min.txt",
                 6 : "logs-c1b1s50-30min.txt",
                 7 : "logs-c1b1s100-30min.txt",
                 8 : "logs-c1b2s100-30min.txt",
                 9 : "logs-c2b1s100-30min.txt",
                 10: "logs-c3b1s100-30min.txt"}

def get_cbs(fname):
    """
    Return cardinality, block, spectral_step parameters from a given log file name
    """
    curr = fname.split('-')[1]
    c = int(curr[1])
    b = int(curr[3])
    s = int(curr[5:])

    return c, b, s

def data_extraction(fname):
    """
    Return lists of train_loss, train_acc, test_loss, test_acc from a given log file.
    """
    train_loss, train_acc, test_loss, test_acc = [], [], [], []

    with open("./plots/"+fname, 'r') as f:
        for line in f:
            curr = line.split(", ")
            train_loss.append(float(curr[2].split(": ")[1]))
            train_acc.append(float(curr[3].split(": ")[1]) * 100)
            test_loss.append(float(curr[4].split(": ")[1]))
            test_acc.append(float(curr[5].split(": ")[1]) * 100)

    return (train_loss, train_acc, test_loss, test_acc)


def data_plot(fname, short=False):
    """
    Plot accuracy and loss plots of training and testing for a given log file.
    If short flag is True, data entries to be plotted will not exceed first 100 epochs.
    """
    a, b, c, d = data_extraction(fname)
    # if short flag is True, truncate data after the first 150 entries, if data size is larger than 150
    if short and len(a) > 100:
        a, b, c, d = a[:100], b[:100], c[:100], d[:100]

    it = [[i] for i in range(len(a))]
    #xmaj_loc = int(math.ceil(len(a) / 15 / 10) * 10) 
    cardi, blk, step = get_cbs(fname)

    fig, ax = plt.subplots()
    plt.title("Accuracies over Epochs. cardi={c}, blks={b}, step={s}.".format(c=cardi, b=blk, s=step))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    p2, = plt.plot(it, b, label="train_acc")
    p4, = plt.plot(it, d, label="test_acc")
    plt.ylim(60, 102)
    plt.legend(handles=[p2, p4])
    ax.grid(True)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.savefig(fname.split('-')[1]+"-acc.png")

    fig, ax = plt.subplots()
    plt.title("Losses over Epochs. cardi={c}, blks={b}, step={s}.".format(c=cardi, b=blk, s=step))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    p1, = plt.plot(it, a, label="train_loss") 
    p3, = plt.plot(it, c, label="test_loss")
    plt.ylim(-0.1, 1.5)
    plt.legend(handles=[p1, p3])
    ax.grid(True)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.02))

    plt.savefig(fname.split('-')[1]+"-loss.png")
    plt.show()

if __name__ == '__main__':
    # for fname in log_file_name.values():
    #     data_plot(fname, True)

    for fname in log_file_name.values():
        with open("./plots/"+fname, 'r') as f:
            c = 1
            for line in f:
                if c != 100:
                    c += 1
                else:
                    with open('logs-100epoch.txt', 'a') as f:
                        f.write(fname+'\n')
                        f.write(line+'\n')
                        break
    

