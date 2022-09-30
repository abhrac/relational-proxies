from utils import constants
from utils import misc
from utils.options import Options


def main():
    args = Options().parse()
    init = misc.Initializers(args)

    init.env()
    args, trainloader, testloader = init.data()

    model = init.modeltype()
    save_path, start_epoch = init.checkpoint()

    if args.eval_only:
        model.test(testloader, epoch=0)
    else:
        for epoch in range(start_epoch + 1, constants.END_EPOCH + 1):
            model.train_one_epoch(trainloader, epoch, save_path)
            model.test(testloader, epoch)

    model.post_job()


if __name__ == "__main__":
    main()
