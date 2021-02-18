# coding=utf-8
import tensorflow as tf


class OdpsData(object):
    def __init__(self, columns, defaults):
        self.columns = columns
        self.defaults = defaults
    
    def get_batch_data(self, tables, slice_id, slice_count, num_epoch, batch_size):
        print('tables: {}, slice_id: {}, slice_count: {}'.format(tables, slice_id, slice_count))
        print('epoch: {}, batch size: {}'.format(num_epoch, batch_size))

        dataset = tf.data.TableRecordDataset(tables,
                                             record_defaults=self.defaults,
                                             slice_id=slice_id,
                                             slice_count=slice_count,
                                             num_threads=4,
                                             capacity=batch_size*10
                                             ).shuffle(batch_size*10).repeat(num_epoch).batch(batch_size).prefetch(10)
        batch_data = dataset.make_one_shot_iterator().get_next()
        return {k: v for k, v in zip(self.columns.split(','), batch_data)}
