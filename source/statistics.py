"""

Bu klasta ilgili istatistikler yer almali ne bileyim iste confussion matrix, accuracy matrix gibi 
seyler hesaplanmali 

Su kadar ADL, FALL instance'i var. 
Su kadar ADL, FALL tipi var.
Su kadar ADL, FALL instance'i var tiplere göre ama. Yani ADL->BSC tipi olsun bundan 5 tane gibi

Bu tarz islemlerin istatistiginin tutuldugu sinif olmasi lazim buranin. En sonunda dosya yazilmasi gerekiyor.

"""

def analyze_results(actuals, output_path):
    ### Positive ---> FALL 
    ### Negative ---> ADL
    
    ### True Positive ---> Actual FALL, also predicted as FALL  
    ### False Positive ---> Actual ADL, but predicted as FALL

    ### True Negative ---> Actual ADL, also predicted as ADL
    ### False Negative ---> Actual FALL, but predicted as ADL
    
    accuracy = calculate_accuracy(actuals)
    precision = calculate_precision(actuals)
    recall = calculate_recall(actuals)
    f1_score = calculate_f1_score(precision, recall)
    
    calculate_confussion_matrix(actuals, output_path, "Confussion_Matrix")
    
    save_statistics(actuals, accuracy, precision, recall, f1_score, output_path)
    
    return

def calculate_accuracy(actuals):
    ### Calculating accuracy, accuracy = (TP + TN) / (TP + FP + TN + FN)
    
    accuracy = (actuals['ADL']['predicted_ADL'] + actuals['FALL']['predicted_FALL']) / (actuals['ADL']['predicted_FALL'] + actuals['FALL']['predicted_ADL'] + actuals['ADL']['predicted_ADL'] + actuals['FALL']['predicted_FALL'])
    
    print("Accuracy of the model: ", accuracy)
        
    return accuracy

def calculate_precision(actuals):
    ### Calculating precision, precision = TP / (TP + FP) tahmin edilen fall'ların ne kadarı gerçekten de fall ?
    
    precision = actuals['FALL']['predicted_FALL'] / (actuals['FALL']['predicted_FALL'] + actuals['ADL']['predicted_FALL'])
    
    print("Precision of the model: ", precision)
    
    return precision

def calculate_recall(actuals):
    ### Calculating the recall, recall = TP / (TP + FN) fall verileri ne kadar doğru tahmin edilmiş ? 
    
    recall = actuals['FALL']['predicted_FALL'] / (actuals['FALL']['predicted_FALL'] + actuals['FALL']['predicted_ADL'])
    
    print("Recall of the model: ", recall)
    
    return recall

def calculate_f1_score(precision, recall):
    ### F1 score
    
    f1_score = (2 * precision * recall) / (precision + recall)
    
    print("F1 score: ", f1_score)
    
    return f1_score

def save_statistics(actuals, accuracy, precision, recall, f1_score, output_path):
    from file_operations import save_json
    
    statistics = { 
            'actual_ADL': actuals['ADL'],
            'actual_FALL': actuals['FALL'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score            
    }
    
    save_json(output_path, 'statistics', statistics)
    
    return

def calculate_confussion_matrix(actuals, output_path, title = "Confussion_Matrix"):
    confussion_matrix = [[0,0],[0,0]]
    
    confussion_matrix[0][0] = actuals['ADL']['predicted_ADL']       ## True Negative
    confussion_matrix[0][1] = actuals['ADL']['predicted_FALL']      ## False Positive

    confussion_matrix[1][0] = actuals['FALL']['predicted_ADL']      ## False Negative   
    confussion_matrix[1][1] = actuals['FALL']['predicted_FALL']     ## True Positive     
        
    from visualization import plot_confussion_matrix 
    
    plot_confussion_matrix(confussion_matrix, output_path, title)