import numpy as np

class LinearInverseProblem():
    def __init__(self, 
                input_data,
                A,
                b=0,
                target_data=None,
                sym_problem=False
                )

        self.A = A
        self.b = b

        self.input_data = input_data
        self.target_data = self.target_data

        self.sample_diameter = []
    
    def get_current_k(self):
        return len(self.sample_diameter)



    def get_wc_kernel(self, get_list=False):
        if get_list:
            wc_list = np.copy(self.sample_diameter)
            for i in range(len(self.sample_diameter)-1):
                wc_list[i+1] = max(wc_list[i], wc_list[i+1]) 

            return wc_list
        else:
            return self.sample_diameter.max()

    def get_av_kernel(self, get_list=False):

        if get_list: 
            return np.accumulate(self.sample_diameter) / np.arange(len(1,len(sample_diameter+1)))

        else:
            return np.mean(k[:k+1])**(1/p)


