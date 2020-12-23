

import numpy as np
import math



def zaa(arg):
    
    """
    The function is responsible for choose a size of the vector, hashing by a part of it and then count the number of zeros
    
    input:
    arg (hexadecimal number): The respective hexadecimal number for hashing
    
    Returns:
    zkey (hash-key): The hashing key
    count (int): The number of zeros
    
    """
    size_bucket=4
    binary_string=str(bin(int((arg.replace('\n','')),16))).replace('0b','')
    hash_key=binary_string[:size_bucket]
    zkey=int(hash_key, 2) 
    fnzeros=binary_string[size_bucket:]
    stop=0
    count=0
    while(stop==0 and count<(len(fnzeros)-1)):
        if(fnzeros[count]=='0'):
            count+=1
        if(fnzeros[count]!='0'):
            stop=1
    return (zkey,count)




def part_hll(entry):
    
    """
    The function creates the buckets using the hashkeys and save the respective values without collision treatment

    
    Input:
    entry (array): An array which contains the hash keys and the counted number of zeros
    
    Returns:
    soma_zeros (int) : The sum of the number of zeros of the array
    nbuckets (int) :   The number of buckets
    
    """
    
    
    campo_viril=-1*np.ones(100000000)
    for n in range (len(entry)):
        [zkey,zvalue]=zaa(entry[n])
        if (zvalue > campo_viril[zkey]):
            campo_viril[zkey]=zvalue
    bucket_hll=campo_viril[campo_viril != -1]    
    soma_zeros=sum(bucket_hll)
    nbuckets=len(bucket_hll)
    return(soma_zeros,nbuckets)



def split_data(data):
    
    """
    The function is responsible to split the huge amount of data we need to treat
    apply the algorithm in part of it and then sum all the partial results from the function PART_HLL and
    give as an output the result of the probably CARDINALITY of the data and the respective ERROR.
    
    Input:
    data (list): the list of hexadecimal numbers
    
    Returns:
    cardinality (float): The probably cardinality of the data
    error (float): The error of the filter
    
    """
    
    quebrado=int(len(data)/1390)
    soma=0
    n=0
    for j in range (1390):
        [soma_aux,n_aux]=part_hll(data[j*quebrado:(j+1)*quebrado])
        soma=soma+soma_aux
        n=n+n_aux
    cardinality=n*(2**(soma/n))
    error=(1.3/math.sqrt(n))*100
    print("Cardinality:",int(cardinality))
    print("Error:", error)
    return (cardinality,error)

