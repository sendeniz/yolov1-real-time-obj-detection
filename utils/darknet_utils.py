import torch 

def top1accuracy(out, target, batch_size):
    """
    Calculates top 1 accuracy.
    Input: Output of class probabilities from the neural network (tensor)
   and target class predictions (tensor) of shape number of classes by batch size 
    Output: Top 1 accuracy (float).
    """
    with torch.no_grad():
        pred_class = torch.argmax(out, dim = 1)
        top1_acc = float(sum(target==pred_class) / batch_size)
        return top1_acc

def top5accuracy(out, target, batch_size):
    """
    Calculates top 1 accuracy.
    Input: Output of class probabilities from the neural network (tensor)
    of shape number of classes by batch size.
    Output: Top 5 accuracy (float).
    """
    with torch.no_grad():
        _, top5_class_pred = out.topk(5, 1, largest = True, sorted = True)
        top5_class_pred = top5_class_pred.t()    
        target_reshaped = target.view(1, -1).expand_as(top5_class_pred)
        correct = (top5_class_pred == target_reshaped)
        ncorrect_top5 = 0
        for i in range(correct.shape[1]):   
            if (sum(correct[:,i]) >= 1):
                ncorrect_top5 = ncorrect_top5 + 1
        top5_acc = ncorrect_top5 / batch_size
        return top5_acc