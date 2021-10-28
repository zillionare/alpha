from . import data
import datetime

def buy_limit_price(code, c0, date: datetime.date):
    """计算相对于c0的收盘价的涨停价

    原则上，涨停板是相对于前一天收盘价的1.10，但由于股价必须取整到分（人民币），因此最终得到的涨停价相对于前一天收盘价，可能并不是10%，而是9.9%，或者10.1%等情况。
    
    由于没有办法判断某只股在`date`时是否为ST股，所以只能判断10%和20%涨停的情况。
    创业板在2020年8月21日收盘后，涨跌停限制由10%变为20%，因此，如果c0对应的交易日是2020年8月21日，则下一个交易日的涨停价格为20%。

    Args:
        code (str): 股票代码
        c0 (float): 当前价格
        date (datetime.date): c0对应的交易日
    """
    if code.startswith('3'):
        if date >= datetime.date(2020, 8, 21):
            threshold = 1.2
        else:
            threshold = 1.1
    elif code.startswith('688'):
        threshold = 1.2
    else:
        threshold = 1.1

    return math_round(c0 * threshold, 2)

def equal_price(x: float, y: float):
    """比较两个股价是否相等。

    如果股价相等，则它们小数点后第二位数字相等，后面的忽略。
    """
    return abs(x - y) < 0.00999

def math_round(x: float, decimal: int):
    """由于浮点数的表示问题，很多语言的round函数与数学上的round函数不一致。下面的函数结果与数学上的一致。
    """
    return int(x * (10 ** decimal) + 0.5) / (10 ** decimal)

