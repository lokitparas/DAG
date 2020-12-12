'''
Function for Dense Adversarial Generation
Adversarial Examples for Semantic Segmentation
Muhammad Ferjad Naeem
ferjad.naeem@tum.de
'''
import torch
from torch.nn import functional as F
import numpy as np
from utils import make_one_hot


def DAG(model,image,ground_truth,adv_target,qual,num_iterations=20,gamma=0.07,no_background=True,background_class=0,device='cuda:0',verbose=False):
    '''
    Generates adversarial example for a given Image
    
    Parameters
    ----------
        model: Torch Model
        image: Torch tensor of dtype=float. Requires gradient. [b*c*h*w]
        ground_truth: Torch tensor of labels as one hot vector per class
        adv_target: Torch tensor of dtype=float. This is the purturbed labels. [b*classes*h*w]
        num_iterations: Number of iterations for the algorithm
        gamma: epsilon value. The maximum Change possible.
        no_background: If True, does not purturb the background class
        background_class: The index of the background class. Used to filter background
        device: Device to perform the computations on
        verbose: Bool. If true, prints the amount of change and the number of values changed in each iteration
    Returns
    -------
        Image:  Adversarial Output, logits of original image as torch tensor
        logits: Output of the Clean Image as torch tensor
        noise_total: List of total noise added per iteration as numpy array
        noise_iteration: List of noise added per iteration as numpy array
        prediction_iteration: List of prediction per iteration as numpy array
        image_iteration: List of image per iteration as numpy array

    '''

    input_sizes = list()
    for img in image:
        size = img.shape[1:]
        input_sizes.append(size)
    print("Input sizes")
    print(input_sizes)

    print(image.shape)
    image.requires_grad_()
    noise_total=[]
    noise_iteration=[]
    prediction_iteration=[]
    image_iteration=[]
    background=None
    _bboxes, _labels, _scores = model.predict(image,visualize=True)
    print("printing label")
    print(_labels)
    # logits=model(image,flag=False)
    # prob = F.softmax(torch.autograd.Variable(logits[1]), dim=1)
    # print("Printing prob")
    # print(prob)
    orig_image=image
    pred_labels = torch.tensor(_labels, dtype=torch.int64)
    pred_labels = pred_labels.to(device)
    # # logits = logits[0]
    # #logits = torch.tensor(logits)
    # print("here")
    # print(logits[0].shape)
    # print(logits[1])

    # _,predictions_orig=torch.max(prob,1)
    # print(predictions_orig)
    # predictions_orig = predictions_orig.unsqueeze(0).unsqueeze(0).cuda()
    
    #TODO: Fix this
    predictions_orig=make_one_hot(pred_labels,21,device)
    
    if(no_background):
        background=torch.zeros(logits.shape)
        background[:,background_class,:,:]=torch.ones((background.shape[2],background.shape[3]))
        background=background.to(device)

    for a in range(num_iterations):
        # output=model(image,flag=False)

        _, output, output_scores = model.predict(image, visualize=True)
        # output = torch.tensor(output)
        # _,predictions=torch.max(output[1],0)
        prediction_iteration.append(np.array(output))
        # predictions=predictions.unsqueeze(0).unsqueeze(0).cuda()
        output = torch.tensor(output,dtype=torch.int64).to(device)
        output_scores = torch.tensor(output_scores,requires_grad=True).to(device)
        predictions=make_one_hot(output,21,device)
        predictions.requires_grad_()
        print(predictions)
        print(ground_truth)
        condition1=torch.eq(predictions,ground_truth).float()
        condition1.requires_grad_()
        condition=condition1
       
        if no_background:
            condition2=(ground_truth!=background)
            condition=torch.mul(condition1,condition2)
        condition=condition.float()
        condition.requires_grad_()

        if(condition.sum()==0):
            print("Condition Reached")
            image=None
            break
        
        #Finding pixels to purturb
        adv_log=torch.mul(output_scores,adv_target)
        #Getting the values of the original output
        clean_log=torch.mul(output_scores,ground_truth)
        print("clean log")
        print(clean_log)
        #Finding r_m
        adv_direction=adv_log-clean_log
        adv_direction.requires_grad_()
        print(adv_direction)
        r_m=torch.mul(adv_direction,condition)
        r_m.requires_grad_()
        print("print rm")
        print(r_m)
        #Summation
        r_m_sum=r_m.sum()
        r_m_sum.requires_grad_()
        print("rm_sum"+str(r_m_sum))
        #Finding gradient with respect to image
        r_m_grad=torch.autograd.grad(outputs=r_m_sum,inputs=image,retain_graph=True)
        print("Grad")
        print(r_m_grad)
        #Saving gradient for calculation
        r_m_grad_calc=r_m_grad[0]
        
        #Calculating Magnitude of the gradient
        r_m_grad_mag=r_m_grad_calc.norm()
        
        if(r_m_grad_mag==0):
            print("Condition Reached, no gradient")
            #image=None
            break
        #Calculating final value of r_m
        r_m_norm=(gamma/r_m_grad_mag)*r_m_grad_calc

        #if no_background:
        if False:
            condition_image=condition.sum(dim=1)
            condition_image=condition_image.unsqueeze(1)
            r_m_norm=torch.mul(r_m_norm,condition_image)

        #Updating the image
        image=torch.clamp((image+r_m_norm),0,1)
        image_iteration.append(image[0][0].detach().cpu().numpy())
        noise_total.append((image-orig_image)[0][0].detach().cpu().numpy())
        noise_iteration.append(r_m_norm[0][0].cpu().numpy())

        if verbose:
            print("Iteration ",a)
            print("Change to the image is ",r_m_norm.sum())
            print("Magnitude of grad is ",r_m_grad_mag)
            print("Condition 1 ",condition1.sum())
            if no_background:
                print("Condition 2 ",condition2.sum())
                print("Condition is", condition.sum()) 

    return image, logits, noise_total, noise_iteration, prediction_iteration, image_iteration
