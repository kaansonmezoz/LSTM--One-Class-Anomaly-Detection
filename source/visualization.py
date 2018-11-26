import numpy as np
import matplotlib.pyplot as plt


def plot_ADL_validation(predicted, lower_threshold, upper_threshold, OUTPUT_PATH):
    # prediction of validation test which is used for determining the threshold value
    
    x = range(len(predicted))
    
    upper_threshold_line = {'x': x, 'y': np.full(len(predicted), upper_threshold)}
    lower_threshold_line = {'x': x, 'y': np.full(len(predicted), lower_threshold)}
    
    plt.plot(predicted)
    plt.plot(upper_threshold_line['x'], upper_threshold_line['y'],color = 'red')
    plt.text(len(predicted), upper_threshold, 'y = ' + str(upper_threshold))
    plt.plot(lower_threshold_line['x'], lower_threshold_line['y'], color = 'red')
    plt.text(len(predicted), lower_threshold, 'y = ' + str(lower_threshold))
    plt.title('prediction values of ADL validation set')
    plt.ylabel('output')
    plt.xlabel('input window no')
    plt.legend(['predicted output', 'threshold'], loc='upper left')
    plt.savefig(OUTPUT_PATH + '/' + 'prediction_validation_set_ADL.png')
    plt.show()
    
    return

def plot_ADL_test(predicted, lower_threshold, upper_threshold, OUTPUT_PATH):
    
    x = range(len(predicted))
    
    upper_threshold_line = {'x': x, 'y': np.full(len(predicted), upper_threshold)}
    lower_threshold_line = {'x': x, 'y': np.full(len(predicted), lower_threshold)}
    
    plt.plot(predicted)
    plt.plot(upper_threshold_line['x'], upper_threshold_line['y'],color = 'red')
    plt.text(len(predicted), upper_threshold, 'y = ' + str(upper_threshold))
    plt.plot(lower_threshold_line['x'], lower_threshold_line['y'], color = 'red')
    plt.text(len(predicted), lower_threshold, 'y = ' + str(lower_threshold))
    plt.title('prediction values of ADL test set')
    plt.ylabel('output')
    plt.xlabel('input window no')
    plt.legend(['predicted output', 'threshold'], loc='best')
    plt.savefig(OUTPUT_PATH + '/' + 'prediction_test_set_ADL.png')
    plt.show()
    
    return

def plot_FALL_test(predicted, lower_threshold, upper_threshold, OUTPUT_PATH, fig_name = 'prediction_test_set_FALL'):
    
    x = range(len(predicted))
    
    upper_threshold_line = {'x': x, 'y': np.full(len(predicted), upper_threshold)}
    lower_threshold_line = {'x': x, 'y': np.full(len(predicted), lower_threshold)}
    
    plt.plot(predicted)
    plt.plot(upper_threshold_line['x'], upper_threshold_line['y'],color = 'red')
    plt.text(len(predicted), upper_threshold, 'y = ' + str(upper_threshold))
    plt.plot(lower_threshold_line['x'], lower_threshold_line['y'], color = 'red')
    plt.text(len(predicted), lower_threshold, 'y = ' + str(lower_threshold))
    plt.title('prediction values of FALL test set')
    plt.ylabel('output')
    plt.xlabel('input window no')
    plt.legend(['predicted output', 'threshold'], loc='best')
    plt.savefig(OUTPUT_PATH + '/' + fig_name +'.png')
    plt.show()
    
    return

def plot_loss(history, OUTPUT_DIRECTORY, OUTPUT_MODEL_NAME, params):
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(range(0, params['epochs'] + 1, 5))
    ##plt.yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    #plt.yticks(i for i in np.arrange(0, history.history['loss'].max + 0.0001, 0.05))
    plt.yticks(create_yticks_array(min(history.history['loss']), max(history.history['loss'])))
    plt.legend(['train'], loc='upper left')
    plt.savefig(OUTPUT_DIRECTORY + '/' + OUTPUT_MODEL_NAME + '/' + 'model-loss.png')
    plt.show()
    
    return

def create_yticks_array(min_value, max_value):
    yticks = []
    
    for i in np.arange(min_value, max_value + 0.0001, 0.05):
        yticks.append(i)
        
    return yticks
