from utils import constants
from utils import misc
from utils.options import Options


def main():
    args = Options().parse()
    init = misc.Initializers(args)

    init.env()
    args, trainloader, testloader = init.data()

    model = init.params()
    save_path, start_epoch = init.checkpoint()

    for epoch in range(start_epoch + 1, constants.END_EPOCH + 1):
        model.train_one_epoch(trainloader, epoch, save_path)
        if epoch % constants.TEST_EVERY == 0:
            model.test(testloader, epoch, save_path)

    model.post_job()


if __name__ == "__main__":
    main()
