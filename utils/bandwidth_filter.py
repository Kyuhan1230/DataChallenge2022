# -*- coding: utf-8 -*-
import copy


def _check_minus(data):
    """
    음수 여부를 체크하여 음수일 경우 0으로 반환한다.
    Args:
        data (float): 체크 대상 데이터

    Returns:
        0 if data < 0, data otherwise
    """
    if data < 0:
        return 0
    else:
        return data


def _set_x_status(data_size: int, y_num: int):
    """
    X Status 리스트를 생성한다.
    Args:
        data_size (int): 총 데이터 개수
        y_num (int): y 변수 개수

    Returns:
        x_status
    """
    x_num = int(data_size - y_num)
    return ["정상" for i in range(x_num)]


def filter_data(raw_data, tag_desc, y_num: int):
    """
    Bandwidth Filter로 데이터를 전처리한다.
    데이터가 경계값 이상일 경우 High를 
    데이터가 경계값 이하일 경우 Low를 반환한다.
     
    Args:
        raw_data: 전체 데이터 
        tag_desc: tag description 및 bwf가 있는 table[(),(),()]
        y_num: y변수 개수

    Returns:
        filtered: 전처리된 데이터
    """
    filtered = copy.deepcopy(raw_data)
    x_status = _set_x_status(data_size=len(raw_data), y_num=y_num)
    
    for i in range(len(tag_desc) - y_num):
        if tag_desc[i][2] == "O":
            high = tag_desc[i][3]
            low = tag_desc[i][4]

            if low is None and high is None:
                pass

            elif low is None:
                if filtered[i] > float(high):
                    filtered[i] = float(high)
                    x_status[i] = "비정상"

            elif high is None:
                if filtered[i] < float(low):
                    filtered[i] = float(low)
                    x_status[i] = "비정상"

            else:
                if filtered[i] > float(high):
                    filtered[i] = float(high)
                    x_status[i] = "비정상"

                elif filtered[i] < float(low):
                    filtered[i] = float(low)
                    x_status[i] = "비정상"

    return [filtered, x_status]
