import torch
import numpy as np
from tqdm.auto import tqdm


def affine_sample(log_prob, n_steps, current_state, args=[], progressbar=True):
    # split the current state
    current_state1, current_state2 = current_state
        
    # pull out the number of parameters and walkers
    n_walkers, n_params = current_state1.shape

    # initial target log prob for the walkers (and set any nans to -inf)...
    logp_current1 = log_prob(current_state1, *args)
    logp_current2 = log_prob(current_state2, *args)
    logp_current1 = torch.where(torch.isnan(logp_current1), torch.ones_like(logp_current1)*torch.log(torch.tensor(0.)), logp_current1)
    logp_current2 = torch.where(torch.isnan(logp_current2), torch.ones_like(logp_current2)*torch.log(torch.tensor(0.)), logp_current2)
    
    chain = [torch.unsqueeze(torch.concat([current_state1, current_state2], dim=0), dim=0)]
    
    for epoch in tqdm(range(1, n_steps), desc=f"{n_params} parameters, {n_walkers} walkers"):
        

        partners1 = torch.tensor(current_state2[np.random.randint(0, n_walkers, (n_walkers,))])
        z1 = 0.5 * (torch.rand(n_walkers) + 1) ** 2
        proposed_state1 = partners1 + (z1*(current_state1 - partners1).T).T
        
        # target log prob at proposed points
        logp_proposed1 = log_prob(proposed_state1, *args)
        logp_proposed1 = torch.where(torch.isnan(logp_proposed1), torch.ones_like(logp_proposed1)*torch.log(torch.tensor(0.)), logp_proposed1)
        
        # acceptance probability
        p_accept1 = torch.minimum(torch.ones(n_walkers), z1**(n_params-1)*torch.exp(logp_proposed1 - logp_current1) )
        
        # accept or not
        accept1_ = ((torch.rand(n_walkers)) <= p_accept1)
        accept1 = accept1_.to(torch.float32)
        
                # update the state
        current_state1 = ( ((current_state1).T)*(1-accept1) + ((proposed_state1).T)*accept1).T
        logp_current1 = torch.where(accept1_, logp_proposed1, logp_current1)

        # second set of walkers:

        # proposals
        partners2 = torch.tensor(current_state1[np.random.randint(0, n_walkers, (n_walkers,))])
        z2 = 0.5 * (torch.rand(n_walkers) + 1) ** 2
        proposed_state2 = partners2 + (z2*(current_state2 - partners2).T).T
        
        # target log prob at proposed points
        logp_proposed2 = log_prob(proposed_state2, *args)
        logp_proposed2 = torch.where(torch.isnan(logp_proposed2), torch.ones_like(logp_proposed2)*torch.log(torch.tensor(0.)), logp_proposed2)
        
        # acceptance probability
        p_accept2 = torch.minimum(torch.ones(n_walkers), z2**(n_params-1)*torch.exp(logp_proposed2 - logp_current2) )
        
        # accept or not
        accept2_ = ((torch.rand(n_walkers)) <= p_accept2)
        accept2 = accept2_.to(torch.float32)
        
        # update the state
        current_state2 = ( ((current_state2).T)*(1-accept2) + ((proposed_state2).T)*accept2).T
        logp_current2 = torch.where(accept2_, logp_proposed2, logp_current2)
        
        # append to chain
        chain.append(torch.unsqueeze(torch.concat([current_state1, current_state2], dim=0), dim=0))
        
        
    return torch.concat(chain, dim=0)

        


        
