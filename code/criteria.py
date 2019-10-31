import numpy as np
from chainer import Variable
import chainer.functions as functions
import math
import random


def mre(effort, effort_hat):
    mre_value = abs((effort.reshape(-1, 1) - effort_hat.reshape(-1, 1))/effort) * 100
    return mre_value


def mre_gpu(effort, effort_hat):
    effort = Variable(effort)
    effort_hat = Variable(effort_hat)
    mre_value = functions.absolute((effort - effort_hat) / effort) * 100
    return mre_value


def mmre(effort, effort_hat):
    mre_value = mre(effort, effort_hat)
    mmre_value = sum(mre_value) / len(mre_value)
    return mmre_value


def mdmre(effort, effort_hat):
    mre_value = mre(effort, effort_hat)
    mdmre_value = np.median(mre_value)
    return mdmre_value


def rsd(effort, effort_hat, afp):
    rsd_value = math.sqrt(np.sum(np.square(np.divide(effort-effort_hat, afp.astype('float32'))))/(len(effort)-1))
    return rsd_value


def lsd(effort, effort_hat):
    e = np.log(effort) - np.log(effort_hat)
    s2 = np.var(e)
    """
    cha = e - (-1/2 * s2)
    square = np.square(cha)
    sum = np.sum(square)
    divide = sum / (len(e)-1)
    sqrt = math.sqrt(divide)
    """
    lsd_value = math.sqrt(np.sum(np.square(e - (-1/2 * s2)))/(len(e)-1))
    print('LSD:', lsd_value)
    return lsd_value


def pred(effort, effort_hat, which_percent=25):
    mre_value = mre(effort, effort_hat)
    data_len = len(effort)
    percent = which_percent
    count = 0
    for a_mre in mre_value:
        if a_mre <= percent:
            count += 1
    pred_value = 100.0/data_len*count
    return pred_value


def pred_gpu(effort, effort_hat, data_size, which_percent=25):
    mre_value = mre_gpu(effort, effort_hat)
    data_len = data_size
    percent = which_percent
    count = 0
    for a_mre in mre_value.data:
        if a_mre <= percent:
            count += 1
    pred_value = 100.0 / data_len * count
    return pred_value


def pred25(effort, effort_hat):
    pred25_value = pred(effort, effort_hat, which_percent=25)
    return pred25_value


def pred25_gpu(effort, effort_hat, data_size):
    pred25_value = pred_gpu(effort, effort_hat, data_size, which_percent=25)
    return pred25_value


def mae(effort, effort_hat):
    mae_value = np.mean(np.fabs(effort.reshape(-1, 1) - effort_hat.reshape(-1, 1)))
    return mae_value


def mar(effort, effort_hat):
    sum_all = np.sum(np.fabs(effort.reshape(-1, 1) - effort_hat.reshape(-1, 1)))
    length = len(effort)
    mar_value = sum_all / length
    return mar_value


def mar_p_zero(effort):
    sum_all = 0.0
    for i in range(1000):
        tmp = effort.copy()
        i_index = random.randint(0, len(effort)-1)
        tmp = np.delete(tmp, i_index)
        index = random.randint(0, len(tmp)-1)
        sum_all += abs(tmp[index]-effort[i_index])
    result = sum_all/1000
    return result


def sa(effort, effort_hat):
    mar_pi = mar(effort, effort_hat)
    mar_p0 = mar_p_zero(effort_hat)
    result = (1-(mar_pi/mar_p0))*100
    return np.float32(result[0])


def re(effort, effort_hat):
    return np.divide(np.var(abs((effort - effort_hat))), np.var(effort_hat))


if __name__ == "__main__":
    effort_value = np.asarray([23, 56, 32, 79])
    effort_hat_value = np.asarray([25, 48, 45, 90])
    afp_value = np.asarray([100, 100, 100, 100])
    # lsd = lsd(effort, effort_hat)
    # print('mre', mre(effort_value, effort_hat_value))
    # print('mmre', mmre(effort_value, effort_hat_value))
    # print('mdmre', mdmre(effort_value, effort_hat_value))
    # print('pred25', pred25(effort, effort_hat))
    # print('lsd', lsd)
    # print('mae', mae(effort_value, effort_hat_value))
    # print('mar', mar(effort_value, effort_hat_value))
    # re_value = re(effort_value, effort_hat_value)
    # print(re_value)
    # print('zero', mar_p_zero(effort_value))
    print('sa', sa(effort_value, effort_hat_value))
