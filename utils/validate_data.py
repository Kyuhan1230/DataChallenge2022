import numpy as np

from scipy.stats import wasserstein_distance

def validate_data(train_data, test_data):
    """
    TFDV와 같은 라이브러리를 이용하면 데이터 검증을 손쉽게 할 수 있으나, 
    제조 공정에 바로 적용할 수 있는 데이터 파이프라인이 없다고 가정하고
    하드 코딩을 통해 데이터 검증을 구현하고자 한다.
    """
    a = train_data
    b = test_data

    ew_dist = round(wasserstein_distance(train_data, test_data),2)
    # 평균 차이 계산
    mean_diff = round(abs(train_data.mean() - test_data.mean()), 2)
    # psi_t = calculate_psi(a, b)
    low = train_data.mean() - ew_dist*0.5
    high = train_data.mean() + ew_dist*0.5
    
    if low < test_data.mean() < high :
        return True
    else:
        print("Data is Changed")
        
        fig, ax = plt.subplots(figsize=(12,4))
        ax.hist(train_data, color='k', weights=np.ones(len(train_data)) / len(train_data), 
                alpha=0.5, label='Train', bins=50)
        ax.hist(test_data, color='r', weights=np.ones(len(test_data)) / len(test_data) ,
                alpha=0.5, label='Test', bins=50)
        ax.legend(loc='upper right');
        plt.show();
        
        hap =np.concatenate((train_data, test_data), axis=0)
        
        plt.figure(figsize=(20,4))
        plt.axvline(x=len(train_data), c='r')
        plt.plot(hap)
        plt.show();
        return False
