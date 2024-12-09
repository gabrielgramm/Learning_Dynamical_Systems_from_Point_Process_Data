import torch
import numpy as np
from helpers.helpers import Helper
from helpers.helpers import no_grad_method

class Kullback_Leibler:

    def kl_lmbda(a_0, b_0 , a_post , b_post):
        '''
        temp1 = (a_0 - a_post) * torch.digamma(a_0)
        temp2 = torch.lgamma(a_0) - torch.lgamma(a_post)
        temp3 = a_post * (torch.log(b_0) - torch.log(b_post))
        temp4 = a_0 * ((b_post - b_0) / b_0)
        kl = temp1 - temp2 + temp3 + temp4
        
        print("seperate terms of the lmbda kl:", temp1.item(), -temp2.item(), temp3.item(), temp4.item())
        print("used lmbda kl", kl.item())
        
        #other direction
        temp1 = (a_post - a_0) * torch.digamma(a_post)
        temp2 = torch.lgamma(a_post) - torch.lgamma(a_0)
        temp3 = a_0 * (torch.log(b_post) - torch.log(b_0))
        temp4 = a_post * ((b_0 - b_post) / b_post)
        kl2 = temp1 - temp2 + temp3 + temp4
        print("lmbda kl other direction", kl2.item())
        '''
        #old KL but without error
        temp1 = (a_post - a_0) * torch.digamma(a_post)
        temp2 = torch.lgamma(a_post) - torch.lgamma(a_0)
        temp3 = a_0 * (torch.log(b_post) - torch.log(b_0))
        temp4 = (b_post - b_0) * (a_post / b_post)
        kl3 = temp1 - temp2 + temp3 - temp4
        '''
        #old other direction
        temp1 = (a_0 - a_post) * torch.digamma(a_0)
        temp2 = torch.lgamma(a_0) - torch.lgamma(a_post)
        temp3 = a_post * (torch.log(b_0) - torch.log(b_post))
        temp4 = (b_0 - b_post) * (a_0 / b_0)
        kl4 = temp1 - temp2 + temp3 - temp4
        print(temp1.item(), temp2.item(), temp3.item(), temp4.item())
        print("kl old other direction", kl4)
        '''
        return kl3
    
    def kl_omega_N(time_discretization, c_n, c_n_squard, E_omega_N):
        temp1 = torch.log(torch.cosh(c_n / 2))
        temp2 = (c_n_squard / 2) * E_omega_N
        kl = sum(temp1 - temp2)
        return kl
    
    def kl_marked_process(end_time, time_discretization, marked_rate, E_lmbda, E_ln_lmbda, c_complete, c_squard_complete,E_omega_complete):
        integrand = 1 + E_ln_lmbda
        temp1 = torch.sum(marked_rate * integrand) / time_discretization
        temp2 = end_time * E_lmbda
        ln_rate = torch.log(marked_rate)
        temp3 = torch.sum(ln_rate * marked_rate) / time_discretization
        integrand = torch.log(torch.cosh(c_complete / 2)) * marked_rate
        temp4 = torch.sum(integrand) / time_discretization
        integrand = c_squard_complete/2 * E_omega_complete * marked_rate
        temp5 = torch.sum(integrand) / time_discretization
        kl = - temp1 + temp2 + temp3 + temp4 - temp5
        return  kl
    
    def kl_gp(mu_s_0, mu_post, cov_prior, cov_post, K_ss, inv_Kss):
        D = mu_s_0.shape[0]
        diff = mu_s_0 - mu_post

        # kl is made of three terms
        tr_term   = torch.sum(inv_Kss * cov_post.mT)
        #tr_term2 = torch.trace(torch.matmul(inv_Kss, cov_post))
        #Helper.test_is_same(tr_term, tr_term2) # --> was true
        quad_term = torch.dot(diff, torch.matmul(inv_Kss , diff)) #np.sum( (diff*diff) * iS1, axis=1)
        log_det_cov_prior = Helper.logdet(cov_prior)
        log_det_cov_post = Helper.logdet(cov_post)
        det_term = log_det_cov_prior - log_det_cov_post

        kl = 0.5 * (tr_term - D + quad_term + det_term)

        ''' reverse '''
        '''
        inv_cov_post = torch.inverse(cov_post)
        diff = mu_post - mu_s_0
        tr_term = torch.sum(inv_cov_post * K_ss.mT)
        quad_term = torch.dot(diff, torch.matmul(inv_cov_post , diff)) #np.sum( (diff*diff) * iS1, axis=1)
        log_det_cov_prior = Helper.logdet(cov_prior)
        log_det_cov_post = Helper.logdet(cov_post)
        det_term = log_det_cov_post - log_det_cov_prior 
        kl2 = 0.5 * (tr_term - D + quad_term + det_term)
        '''
        return kl
        

    
