import numpy as np
import matplotlib.pyplot as plt


def plot_set_results(predicted, lower_threshold, upper_threshold, OUTPUT_PATH, title, fig_name, loc): ### ADL_validation icin loc='upper left', geri kalan datasetleri icin loc='best' olmali ama bir bakalim gene de diger turlu sekillerde de denemek lazim
    # prediction of validation test which is used for determining the threshold value
    
    x = range(len(predicted))
    
    upper_threshold_line = {'x': x, 'y': np.full(len(predicted), upper_threshold)}
    lower_threshold_line = {'x': x, 'y': np.full(len(predicted), lower_threshold)}
    
    plt.plot(predicted)
    plt.plot(upper_threshold_line['x'], upper_threshold_line['y'],color = 'red')
    plt.text(len(predicted), upper_threshold, 'y = ' + str(upper_threshold))
    plt.plot(lower_threshold_line['x'], lower_threshold_line['y'], color = 'red')
    plt.text(len(predicted), lower_threshold, 'y = ' + str(lower_threshold))
    plt.title(title)    ### prediction values of ADL test set, prediction values of ADL validation set, prediction values of FALL test set
    plt.ylabel('output')
    plt.xlabel('input window no')
    plt.legend(['predicted output', 'threshold'], loc = loc)
    plt.savefig(OUTPUT_PATH + '/' + fig_name + '.png')
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

def plot_confussion_matrix(cm, output_path, title, color_map = plt.cm.Blues):
    if max(cm[0]) > max(cm[1]):
        thres = max(cm[0]) / 2
    else:
        thres = max(cm[1]) / 2
        
    plt.imshow(cm, interpolation = 'nearest', cmap = color_map)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(2) ## we have got only 2 classes
    plt.xticks(tick_marks, ['ADL', 'FALL'], rotation = 45)
    plt.yticks(tick_marks, ['ADL', 'FALL'])
        
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i][j], 'd'), 
                 horizontalalignment = "center", color= "white" if cm[i][j] > thres  else "black")
        
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    plt.savefig(output_path + '/' + title + '.png')
    plt.show()
    
    return    
