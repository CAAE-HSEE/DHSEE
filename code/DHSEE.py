from __future__ import print_function

import time

import chainer
import chainer.functions as f
import chainer.links as links
import math
import numpy as np
import pandas
from chainer import initializers, serializers
from chainer.dataset import convert

from experiment import criteria, read_data_validation


# model
class EncoderRegressionModel(chainer.Chain):
    def __init__(self, in_size=[1, 2], encoder_n_units=10, regression_n_units=10, common_out_size=10):
        super(EncoderRegressionModel, self).__init__()
        with self.init_scope():
            initial_w = initializers.HeNormal()
            self.common_out_size = common_out_size

            # Encoder 1
            self.encoder11 = links.Linear(in_size[0], encoder_n_units, initialW=initial_w)
            self.encoder12 = links.Linear(encoder_n_units, encoder_n_units, initialW=initial_w)
            self.encoder13 = links.Linear(encoder_n_units, encoder_n_units, initialW=initial_w)
            self.encoder14 = links.Linear(encoder_n_units, common_out_size, initialW=initial_w)

            # Encoder 2
            self.encoder21 = links.Linear(in_size[1], encoder_n_units, initialW=initial_w)
            self.encoder22 = links.Linear(encoder_n_units, encoder_n_units, initialW=initial_w)
            self.encoder23 = links.Linear(encoder_n_units, encoder_n_units, initialW=initial_w)
            self.encoder24 = links.Linear(encoder_n_units, common_out_size, initialW=initial_w)

            # Decoder 1
            self.decoder11 = links.Linear(common_out_size, encoder_n_units, initialW=initial_w)
            self.decoder12 = links.Linear(encoder_n_units, encoder_n_units, initialW=initial_w)
            self.decoder13 = links.Linear(encoder_n_units, encoder_n_units, initialW=initial_w)
            self.decoder14 = links.Linear(encoder_n_units, in_size[0], initialW=initial_w)

            # Regression 2
            self.regression21 = links.Linear(common_out_size, regression_n_units, initialW=initial_w)
            self.regression22 = links.Linear(regression_n_units, regression_n_units, initialW=initial_w)
            self.regression23 = links.Linear(regression_n_units, regression_n_units, initialW=initial_w)
            self.conv = links.ConvolutionND(ndim=1, in_channels=1, out_channels=regression_n_units,
                                            ksize=common_out_size, stride=1, pad=0, initialW=initial_w)
            self.fc = links.Linear(encoder_n_units, 1)

    def encoder1_forward(self, x):
        # Encoder1
        h_encoder_11 = f.leaky_relu(self.encoder11(x))
        h_encoder_12 = f.leaky_relu(self.encoder12(h_encoder_11))
        h_encoder_13 = f.leaky_relu(self.encoder13(h_encoder_12))
        encoder_1_output = f.leaky_relu(self.encoder14(h_encoder_13))
        return encoder_1_output

    def encoder2_forward(self, x):
        # Encoder2
        h_encoder_21 = f.leaky_relu(self.encoder21(x))
        h_encoder_22 = f.leaky_relu(self.encoder22(h_encoder_21))
        h_encoder_23 = f.leaky_relu(self.encoder23(h_encoder_22))
        encoder_2_output = f.leaky_relu(self.encoder24(h_encoder_23))
        return encoder_2_output

    def decoder_forward(self, x):
        # Regression 1
        h_decoder_11 = f.leaky_relu(self.decoder11(x))
        h_decoder_12 = f.leaky_relu(self.decoder12(h_decoder_11))
        h_decoder_13 = f.leaky_relu(self.decoder13(h_decoder_12))
        decoder_1_output = f.leaky_relu(self.decoder14(h_decoder_13))
        return decoder_1_output

    def regression2_forward(self, x):
        # Regression 2
        h_regression_21 = f.leaky_relu(self.regression21(x))
        h_regression_22 = f.leaky_relu(self.regression22(h_regression_21))
        h_regression_23 = f.leaky_relu(self.regression23(h_regression_22))
        h1 = f.leaky_relu(h_regression_23)
        h2 = self.fc(h1)
        return h2

    def forward_1(self, x1):
        encoder_1_output = self.encoder1_forward(x1)
        regression_1_output = self.decoder_forward(encoder_1_output)
        return regression_1_output

    def forward_2(self, x2):
        encoder_2_output = self.encoder2_forward(x2)
        encoder_2_output = chainer.Variable(
            encoder_2_output.data.reshape(len(encoder_2_output), 1, self.common_out_size))
        regression_2_output = self.regression2_forward(encoder_2_output)
        return regression_2_output

    def __call__(self, x1, x2):
        encoder_1_output = self.encoder1_forward(x1)
        decoder_1_output = self.decoder_forward(encoder_1_output)

        encoder_2_output = self.encoder2_forward(x2)
        encoder_2_output = chainer.Variable(encoder_2_output.data.reshape(len(encoder_2_output), 1, self.common_out_size))
        regression_2_output = self.regression2_forward(encoder_2_output)

        return encoder_1_output, encoder_2_output, decoder_1_output, regression_2_output


# class
class Discriminator(chainer.Chain):
    def __init__(self, in_size, n_units=10):
        super(Discriminator, self).__init__()
        with self.init_scope():
            initial_w = initializers.HeNormal()
            self.l1 = links.Linear(in_size, n_units, initialW=initial_w)  # n_in -> n_units
            self.l2 = links.Linear(n_units, n_units, initialW=initial_w)
            self.l3 = links.Linear(n_units, 1)

    def __call__(self, x):
        h1 = f.sigmoid(self.l1(x))
        h2 = f.sigmoid(self.l2(h1))
        y = f.sigmoid(self.l3(h2))
        return y


# ---------------------------------------------------key function-----------------------------------------------------------
def main(encoder_n_units=32, common_size=16, regression_n_units=32, discriminator_n_units=32, batch_size_def=None,
         epoch=3000, data_set_name=None, validation_patience_original=1000, train_size=0.7,
         save_code=False, pre_best = [0, 0, 0, 0], target_data_set=None):

    if batch_size_def is None:
        batch_size_def = [6, 20]
    if data_set_name is None:
        data_set_name = ['china', 'kitchenham']
    print("---------------------------------------------Reading data...-----------------------------------------------")
    train = []
    validation = []
    test = []
    in_size = []
    x_train = []
    y_train = []
    x_validation = []
    y_validation = []
    x_test = []
    y_test = []

    for i in range(len(data_set_name)):
        a_name = data_set_name[i]
        a_train, a_validation, a_test, a_in_size, a_x_train, a_y_train, a_x_validation, a_y_validation, a_x_test, \
        a_y_test = read_data_validation.get_train_and_test(dataset=a_name, train_size=train_size, validation_size=0.0)

        train.append(a_train)
        validation.append(a_validation)
        test.append(a_test)
        in_size.append(a_in_size)
        x_train.append(a_x_train)
        y_train.append(a_y_train)
        x_validation.append(x_validation)
        y_validation.append(y_validation)
        x_test.append(a_x_test)
        y_test.append(a_y_test)

    # Prepare the train iter.
    train_iter = []
    for i in range(len(data_set_name)):
        a_train_iter = chainer.iterators.SerialIterator(train[i], batch_size_def[i])
        train_iter.append(a_train_iter)

    # Build model
    print("---------------------------------------Building model...---------------------------------------------------")
    model = EncoderRegressionModel(in_size=in_size, encoder_n_units=encoder_n_units,
                                   regression_n_units=regression_n_units,
                                   common_out_size=common_size)
    model_optimizer = chainer.optimizers.Adam()
    model_optimizer.setup(model)
    # Build Discriminator
    discriminator = Discriminator(common_size, discriminator_n_units)
    discriminator_optimizer = chainer.optimizers.SGD(lr=0.001)
    discriminator_optimizer.setup(discriminator)

    # --------------------------------Measures: pred、MdAE、SA、RE*-----------------------------------------------------
    def test_measure(data_index, which_measure):
        x_array, t_array = convert.concat_examples(test[data_index])
        x = chainer.Variable(x_array)
        y_test_predict = model.forward_2(x).data
        if data_set_name[data_index] in ['cocnas', 'maxwell', 'opens']:
            y_test_predict = np.power(math.e, y_test_predict)
            t_array = np.power(math.e, t_array)

        if which_measure == 'Pred(25)':
            return criteria.pred25(t_array, y_test_predict)
        elif which_measure == 'MdAE':
            return criteria.mae(t_array, y_test_predict)
        elif which_measure == 'SA':
            return criteria.sa(t_array, y_test_predict)
        elif which_measure == 'RE*':
            return criteria.re(t_array, y_test_predict)
        else:
            return None

    def validation_measure(data_index, which_measure):
        x_array, t_array = convert.concat_examples(validation[data_index])
        x = chainer.Variable(x_array)
        y_validation_predict = model.forward_2(x).data
        if data_set_name[data_index] in ['cocnas', 'maxwell', 'opens']:
            y_validation_predict = np.power(math.e, y_validation_predict)
            t_array = np.power(math.e, t_array)

        if which_measure == 'Pred(25)':
            return criteria.pred25(t_array, y_validation_predict)
        elif which_measure == 'MdAE':
            return criteria.mae(t_array, y_validation_predict)
        elif which_measure == 'SA':
            return criteria.sa(t_array, y_validation_predict)
        elif which_measure == 'RE*':
            return criteria.re(t_array, y_validation_predict)
        else:
            return None

    # train
    def discriminator_loss_fun(x1, x2, y1, y2):
        y1_hat = discriminator(x1)
        y1_hat = y1_hat.reshape(len(y1_hat))
        loss1 = f.sigmoid_cross_entropy(y1_hat, y1)
        y2_hat = discriminator(x2)
        y2_hat = y2_hat.reshape(len(y2_hat))
        loss2 = f.sigmoid_cross_entropy(y2_hat, y2)
        loss = loss1 + loss2
        dis_loss.append(loss.data)
        return loss

    def loss_fun(x1, x2, y2, label1, label2):
        encoder_1_output_def, encoder_2_output_def, decoder_1_output, regression_2_output_def = model(x1, x2)
        regression_2_output_def = regression_2_output_def.reshape((len(regression_2_output_def), 1))
        decoder_1_loss = f.mean_absolute_error(decoder_1_output, x1)
        regression_2_loss = f.mean_absolute_error(regression_2_output_def, y2)

        # Generator loss
        y1_hat = discriminator(encoder_1_output_def)
        y1_hat = y1_hat.reshape(len(y1_hat))
        encoder_1_loss = f.sigmoid_cross_entropy(y1_hat, label1)
        y2_hat = discriminator(encoder_2_output_def)
        y2_hat = y2_hat.reshape(len(y2_hat))
        encoder_2_loss = f.sigmoid_cross_entropy(y2_hat, label2)

        loss = decoder_1_loss + regression_2_loss * 2 + encoder_1_loss + encoder_2_loss
        loss_all.append(loss.data)
        # print("Generator loss = ", loss.data)
        return loss

    print("----------------------------------------------------Training...--------------------------------------------")
    chainer.using_config('train', True)
    running = True
    # Storage the loss
    loss_all = []
    dis_loss = []
    validation_frequency = 1
    validation_patience = validation_patience_original
    # Measures: [pred、MdAE、SA、RE*]
    best_validation = [0, 0, 0, 0]
    best_test = [0, 0, 0, 0]

    # running
    while running:
        running_count = 0
        for i in range(len(data_set_name)):
            if train_iter[i].epoch < epoch:
                running_count += 1
            if running_count == 0:
                running = False

        # get batch
        batch1 = train_iter[0].next()
        x_array, t_array = convert.concat_examples(batch1)
        input_x1 = chainer.Variable(x_array)
        input_y1 = chainer.Variable(t_array)

        batch2 = train_iter[1].next()
        x_array, t_array = convert.concat_examples(batch2)
        input_x2 = chainer.Variable(x_array)
        input_y2 = chainer.Variable(t_array)

        # Train Discriminator on the real data
        encoder_1_output, encoder_2_output, regression_1_output, regression_2_output = model(input_x1, input_x2)
        zeros = np.zeros(len(encoder_1_output), dtype=np.int32)
        ones = np.ones(len(encoder_2_output), dtype=np.int32)
        discriminator_optimizer.update(discriminator_loss_fun, encoder_1_output, encoder_2_output, zeros, ones)

        # Train Generator
        zeros = np.zeros(len(encoder_2_output), dtype=np.int32)
        ones = np.ones(len(encoder_1_output), dtype=np.int32)
        model_optimizer.update(loss_fun, input_x1, input_x2, input_y2, ones, zeros)

        # validation
        validation_patience -= 1
        if train_iter[1].epoch % validation_frequency == 0:
            # compute pred25
            validation_pred25 = validation_measure(1, "Pred(25)")
            if validation_pred25 >= best_validation[0]:
                best_validation[0] = validation_pred25
                # test on the test dataset
                test_pred25 = test_measure(1, "Pred(25)")
                if test_pred25 > best_test[0]:
                    best_test[0] = test_pred25
                    # save model
                    if True:
                        serializers.save_npz('../models/multi_' + data_set_name[1] + '.model', model)
                validation_patience = validation_patience_original
        if validation_patience == 0:
            break

    chainer.using_config('train', False)
    print("-------------------------------------- Train finished -----------------------------------------------------")
    print('---------------------------------------------Criteria Test---------------------------------------------')
    data_index = 1
    x_array, t_array = convert.concat_examples(test[data_index])
    x = chainer.Variable(x_array)
    y_predict_data = model.forward_2(x).data

    best_test[0] = criteria.pred25(t_array, y_predict_data)
    best_test[1] = criteria.mae(t_array, y_predict_data)
    best_test[2] = criteria.sa(t_array, y_predict_data)
    best_test[3] = criteria.re(t_array, y_predict_data)
    if best_test[0] > pre_best[0]:
        # Save Code, including the train, validation and test.
        if save_code is True:
            data_index = 0
            x_array, t_array = convert.concat_examples(train[data_index])
            x = chainer.Variable(x_array)
            code_train = model.encoder1_forward(x)
            data1_train_code = code_train.data

            x_array, t_array = convert.concat_examples(validation[data_index])
            x = chainer.Variable(x_array)
            code_validation = model.encoder1_forward(x)
            data1_validation_code = code_validation.data

            x_array, t_array = convert.concat_examples(test[data_index])
            x = chainer.Variable(x_array)
            code_test = model.encoder1_forward(x)
            data1_test_code = code_test.data

            data_index = 1
            x_array, t_array = convert.concat_examples(train[data_index])
            x = chainer.Variable(x_array)
            code_train = model.encoder2_forward(x)
            data2_train_code = code_train.data

            x_array, t_array = convert.concat_examples(validation[data_index])
            x = chainer.Variable(x_array)
            code_validation = model.encoder2_forward(x)
            data2_validation_code = code_validation.data

            x_array, t_array = convert.concat_examples(test[data_index])
            x = chainer.Variable(x_array)
            code_test = model.encoder2_forward(x)
            data2_test_code = code_test.data

            code = np.vstack((data1_train_code, data1_validation_code, data1_test_code, data2_train_code,
                              data2_validation_code, data2_test_code))
            print("Code.shape is ", code.shape)
            gen_data = pandas.DataFrame(code)
            gen_data.to_csv('../data/prevModel.csv')

            return best_test
    else:
        return pre_best


if __name__ == '__main__':
    # list_data = ['china', 'cocnas', 'kitchenham', 'maxwell', 'miyazaki94', 'opens', 'albrecht', 'kemerer']
    list_data = ['kemerer', 'albrecht', 'opens', 'miyazaki94', 'maxwell', 'kitchenham', 'cocnas', 'china']
    batch_size = [50, 15, 15, 15, 15, 15, 5, 10]
    patience = [30, 40, 20, 30, 200, 30, 300, 5]
    pre_best = [-10.0, -10.0, -10.0, -10.0]

    # -----------The structure of the results{target1: {
    #                                   (target1, source1) : measures1,
    #                                   (target1, source2) : measures2,
    #                                   ......},
    #                         target2: {}
    #                         }
    #store the results
    result = dict()

    for index in range(10):
        for target in range(len(list_data)):
            target_result = dict()
            key_str = list_data[target]
            is_first = True
            time_start = time.time()
            source_data = [i for i in list_data if i != list_data[target]]
            source_batch = [batch_size[i] for i in range(len(batch_size)) if i != target]
            source_patience = [patience[i] for i in range(len(patience)) if i != target]
            for source in range(len(source_data)):
                if is_first:
                    pre_best = main(encoder_n_units=32, common_size=16, regression_n_units=32,
                                    discriminator_n_units=32,
                                    batch_size_def=[source_batch[source], source_batch[source]],
                                    epoch=5000, data_set_name=[list_data[target], source_data[source]],
                                    train_size=0.7, save_code=True,
                                    validation_patience_original=source_patience[source], pre_best=pre_best)
                    key_str += (","+source_data[source])
                    target_result[key_str] = pre_best
                    is_first = False
                else:
                    pre_best = main(encoder_n_units=32, common_size=16, regression_n_units=32,
                                    discriminator_n_units=32,
                                    batch_size_def=[source_batch[source], source_batch[source]],
                                    epoch=5000, data_set_name=['prevModel', source_data[source]],
                                    train_size=0.7, save_code=True,
                                    validation_patience_original=source_patience[source], pre_best=pre_best)
                    key_str += ("," + source_data[source])
                    target_result[key_str] = pre_best

            time_end = time.time()
            target_result['time'] = time_end - time_start
            result[list_data[target]] = target_result
            pre_best = [-10.0, -10.0, -10.0, -10.0]
        # iteratively output the results
        write_string = ''
        for target_str, dict_measure in result.items():
            print('-------------------------', target_str, '-----------------------------------')
            write_string += '-------------------------' + target_str + '-----------------------------------\n'
            for key, value in dict_measure.items():
                if key == 'time':
                    print(target_str, "time consuming=", value)
                    write_string += target_str + "time consuming=" + str(value) + '\n'
                else:
                    print(key, ":Pred(25)=", value[0])
                    print(key, ":MdAE=", value[1])
                    print(key, ":SA=", value[2])
                    print(key, ":RE*=", value[3])
                    write_string += key + ":Pred(25)=" + str(value[0]) + '\n'
                    write_string += key + ":MdAE=" + str(value[1]) + '\n'
                    write_string += key + ":SA=" + str(value[2]) + '\n'
                    write_string += key + ":RE*=" + str(value[3]) + '\n'
