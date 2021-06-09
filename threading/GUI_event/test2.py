import os                                                                       
from multiprocessing import Pool
from time import sleep                                             
                                                                                
                                                                                
processes = ('process1.py')                                    
other = ('process2.py',)
                                                  
                                                                                
def run_process(process):                                                             
    os.system('python {}'.format(process))                                       
                                                                                
                                                                                
pool = Pool(processes=2)                                                        
pool.map(run_process, processes)
sleep(5)
pool.map(run_process, other)