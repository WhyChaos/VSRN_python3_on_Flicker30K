from __future__ import print_function
import os
import pickle

import torch
import numpy
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary  # NOQA
from model import VSRN, order_sim
from collections import OrderedDict




class AverageMeter(object):
    """计算和存储当前和平均值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """logging表达
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """日志收集"""

    def __init__(self):
        # 保持日志顺序
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # 如果之前没有记录，则创建一个新仪表
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """记录一条log
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """使用tensorboard记录
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """编码所有的图像文本描述 加载于 `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # 切换至训练模式
    model.val_start()

    end = time.time()

    # numpy数组保持
    img_embs = None
    cap_embs = None
    for i, (images, captions, lengths, ids, caption_labels, caption_masks) in enumerate(data_loader):
        # 确保验证日志被使用
        model.logger = val_logger

        # 计算嵌入
        img_emb, cap_emb, fc_img_emd = model.forward_emb(images, captions, lengths,
                                             volatile=True)

        # 初始化矩阵
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # 通过从 GPU 复制并转换为 numpy 来保留嵌入
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()


        del images, captions

    return img_embs, cap_embs





def evalrank(model_path, model_path2, data_path=None, split='dev', fold5=False):
    """
    在开发或测试中评估经过训练的模型
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    checkpoint2 = torch.load(model_path2)
    opt2 = checkpoint2['opt']

    if data_path is not None:
        opt.data_path = data_path

    #加载词汇表
    with open(os.path.join(opt.vocab_path,
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)



    model = VSRN(opt)
    model2 = VSRN(opt2)

    # 加载模型状态
    model.load_state_dict(checkpoint['model'])

    model2.load_state_dict(checkpoint2['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, data_loader)
    img_embs2, cap_embs2 = encode_data(model2, data_loader)

    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # 没有交叉验证
        r, rt = i2t(img_embs, cap_embs, img_embs2, cap_embs2, measure=opt.measure, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs, img_embs2, cap_embs2, 
                      measure=opt.measure, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5折交叉验证
        results = []
        for i in range(5):
            r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000], cap_embs[i * 5000:(i + 1) *5000], 
                img_embs2[i * 5000:(i + 1) * 5000], cap_embs2[i * 5000:(i + 1) *5000], 
                measure=opt.measure,
                         return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],cap_embs[i * 5000:(i + 1) *5000], 
                img_embs2[i * 5000:(i + 1) * 5000],cap_embs2[i * 5000:(i + 1) *5000], 
                measure=opt.measure,
                           return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def i2t(images, captions, images2, captions2, npts=None, measure='cosine', return_ranks=False):
    """
    图像检索文本
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] / 5
    index_list = []
    npts = int(npts)
    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # 获得询问图像
        im = images[5 * index].reshape(1, images.shape[1])
        im_2 = images2[5 * index].reshape(1, images2.shape[1])
        # C计算分数
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = numpy.dot(im, captions.T).flatten()
            d2 = numpy.dot(im_2, captions2.T).flatten()
            d = (d + d2)/2
            
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # 分数
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # 计算评价
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, images2, captions2, npts=None, measure='cosine', return_ranks=False):
    """
    文本检索图像
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] / 5
    npts = int(npts)
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ims2 = numpy.array([images2[i] for i in range(0, len(images2), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):

        # 获得描述
        queries = captions[5 * index:5 * index + 5]
        queries2 = captions2[5 * index:5 * index + 5]
        # 计算分数
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = numpy.dot(queries, ims.T)
            d2 = numpy.dot(queries2, ims2.T)
            d = (d+d2)/2
            
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # C计算评价
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
