import requests
from PIL import Image
import pytesseract
import numpy as np
from pdf2image import convert_from_path
from docx2pdf import convert

import pickle
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import io
import argparse
from tqdm import tqdm
from googleapiclient.http import MediaIoBaseDownload
import re
import pandas as pd
from datetime import datetime
import cv2
import time
import Levenshtein
from easyocr import Reader

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# tesseract_config = r'--oem 1 --psm 3'
tesseract_config = r'--oem 1 --psm 3 -c tessedit_char_whitelist=0123456789'
reader = Reader(['en'], gpu=True)

logpath = os.path.join(os.getcwd(), 'ocr_script_log.txt')
#outpath = r'C:\Users\govin\Downloads\dst.png'
outpath = os.path.join(os.getcwd(), 'dst.png')

def createLog():
    if os.path.exists(logpath):
        os.remove(logpath)
    f = open(logpath, 'w')


def log(line):
    try:
        with open(logpath, 'a', encoding='utf-8') as f:
            f.write(line+"\n")
    except:
        log('Could not write line'+'\n')

def download_file(url, fpath, fname):
    cwd = os.getcwd()
    destination = os.path.join(cwd, 'Temp')
    move_to = ''

    r = requests.get(url, allow_redirects=True)
    open(fpath, 'wb').write(r.content)

    if os.path.exists(fpath):
        try:
            log(f'Moving {fpath} to Temp folder...')
            move_to = os.path.join(destination, fname)
            # move_to = os.rename(dfilespath, os.path.join(destination, dfilespath))
            os.rename(fpath, move_to)
        except:
            log(f'Could not rename/move file : {fname}')
    log(move_to)
    return move_to

def scan(df1, df2, start):

    for row in df1.itertuples():
        t = datetime.now()
        print(f'Time Elapsed: {t-start}')
        # print(row)
        index = row.Index
        #pdverified = row._32
        #payment3 = str(row._36)

        #log(f'{row._6} {row._7} Payment 3 status: {payment3}')
        #payment2status = row._30
        #if pdverified == 'check':
        #if payment3 == 'nan' or payment3.startswith('HOLD'):
        #if payment2status == r'invoice submitted (2 and 3)':
        full_name = row._2.split()
        firstname = full_name[0]  # ['First Name']
        lastname = full_name[-1]  # ['Last Name']
        email1 = row._13
        email2 = row._14  # ['Email address']
        #print(email2)

        mask1 = df2['Last Name'].values == lastname
        #mask = df2["What is your email address? (We're asking you to provide this so we can send you a copy of your responses. Please use the primary email on your fellow application, all lowercase.)"].values==email
        mask2 = df2['Email'].values == email1
        mask3 = df2['Email'].values == email2
        
        tempdf = df2[mask1]
        mask = mask1
        if tempdf.empty:
            tempdf = df2[mask2]
            mask = mask2
        elif tempdf.empty:
            tempdf = df2[mask3]
            mask = mask3
        log(f'Printing partial DF for {firstname} {lastname}')
        log(f'{tempdf}')

        verified = verifyPD(tempdf, df2, firstname, lastname, mask)
        # row['OCR CHECK'] = 'DONE'
        try:
            df1.at[index, 'OCR CHECK'] = 'DONE'
            log(f'{firstname} {lastname} DONE - DF UPDATED')
        except:
            log(f'{firstname} {lastname} DONE - COULD NOT UPDATE')

    df1.to_csv('out1feb1.csv')
    return

def getIndices(df, mask):
    #mask = df['Last Name'].values == lastname
    #mask = df["What is your email address? (We're asking you to provide this so we can send you a copy of your responses. Please use the primary email on your fellow application, all lowercase.)"].values==email
    #mask = df['Email'].values == email
    tempdf = df[mask]
    indices = []

    for row in tempdf.itertuples():
        indices.append(row.Index)
    return indices


def verifyPD2(df, fulldf, firstname, lastname, mask):
    log(f'Verifying PD for {firstname} {lastname}')
    poppler_path = r'C:\Users\govin\Documents\Poppler\poppler-23.05.0\Library\bin'
    fpath = ''

    indices = getIndices(fulldf, lastname, mask)

    fulldf = mark_duplicates(df, fulldf, firstname, lastname)

    fulldf.to_csv('out2feb1.csv')

def verifyPD(df, fulldf, firstname, lastname, mask):
    cwd = os.getcwd()
    log(f'Verifying PD for {firstname} {lastname}')
    poppler_path = r'C:\Users\govin\Documents\Poppler\poppler-23.05.0\Library\bin'
    fpath = ''

    indices = getIndices(fulldf, mask)

    fulldf = mark_duplicates(df, fulldf, firstname, lastname)

    for idx in indices:
        row = fulldf.iloc[idx]
        #gdrive_link = row['Please upload proof of completion (e.g., certificate)']
        # gdrive_link = row._11

        host_pd = row['HOSTED ORGANIZATION TXT']
        if host_pd=='Other':
            host_pd = row['Please name the organization that hosted this event below.']
        date = convertDate(row['What was the start date of this event?'])
        end_date = convertDate(row['What was the end date of this event?'])
        hours = str(int(row['How many hours of PD credit are you expecting to receive from your participation?']))
        fileurl = row['Please upload your completion certificate here.']

        fname = fileurl[fileurl.rindex('/')+1:].replace(r'%20','_')
        fpath = os.path.join(cwd, 'Downloads', fname)

        fpath = download_file(fileurl, fpath, fname)

        try:
            ocr_check, message = verifyOCR(
                fpath, poppler_path, firstname, lastname, hours, date)
            log('yes1')
            if ocr_check:
                fulldf.at[idx, 'OCR CHECK'] = message
                log('yes2')
            else:
                fulldf.at[idx, 'OCR CHECK'] = message

            log(f'Removing {fpath}...')
            os.remove(fpath)
        except:
            log(f'Could not process file {fname}')
    fulldf.to_csv('out2feb1.csv')


def convertDate(datestring):
    date_object = ''
    try:
        date_object = datetime.strptime(datestring, '%m/%d/%Y')
    except:
        print()

    try:
        date_object = datetime.strptime(datestring, '%m-%d-%Y')
    except:
        print()

    try:
        date_object = datetime.strptime(datestring, '%B %d, %Y')
    except:
        print()

    if date_object != '':
        converted_date = date_object.strftime('%B %d, %Y')
    else:
        converted_date = ''
    return converted_date


def extract_text_from_pdf(pdf_path, poppler_path):
    # Convert PDF to image
    pages = convert_from_path(pdf_path, poppler_path=poppler_path)

    # Extract text from each page using Tesseract OCR
    textdata = ''
    for page in pages:
        image = convertBGR2RGB(page)
        results = reader.readtext(image, mag_ratio=2)
        #text = pytesseract.image_to_string(
        #    page, config=tesseract_config, lang='eng')
        for (bbox, text, prob) in results:
            text = cleanup_text(text)
            textdata += text + '\n'

    # Return the text data
    return textdata


def imgProc(img, outpath):
	image = img
	#image = convertBGR2RGB(image)
	#image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	blured1 = cv2.medianBlur(image,3)
	blured2 = cv2.medianBlur(image,51)
	divided = np.ma.divide(blured1, blured2).data
	normed = np.uint8(255*divided/divided.max())
	#th, threshed = cv2.threshold(normed, 100, 255, cv2.THRESH_OTSU)
	th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                    cv2.THRESH_BINARY,11,2)
	dst = np.vstack((image, blured1, blured2, normed, th3))
	cv2.imwrite(outpath, dst)
	return th3, normed

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def convertBGR2RGB(img):
    imgpath = os.path.join(os.getcwd(), 'page.jpg')
    img.save(imgpath)
    image = cv2.imread(imgpath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    os.remove(imgpath)
    return rgb

def convert_to_img(fpath, poppler_path):
    if fpath.endswith('.docx'):
        convert(fpath)
        fpath = fpath.replace('docx', 'pdf')
    
    pages = convert_from_path(fpath, poppler_path=poppler_path)
    return pages

def extract_datestring_from_ocr_text(line, date):
    months = ['January', 'Jan', 'February', 'Feb', 'March', 'Mar', 'April', 'Apr', 'May', 'June', 'Jun',
              'July', 'Jul', 'August', 'Aug', 'September', 'Sep', 'Sept', 'October', 'Oct', 'November', 'Nov', 'December', 'Dec']
    months_shortform = {'Jan':'January', 'Feb':'February', 'Mar':'March', 'Apr':'April', 'Jun':'June', 'Jul':'July', 'Aug':'August', 'Sep':'September', 'Sept':'September',
                         'Oct':'October', 'Nov':'November', 'Dec':'December'}
    datestart_idx = -1
    dateend_idx = -1
    datestr = ''
    flag_month = False

    for month in months:
        datestart_idx = line.find(month)
        if datestart_idx > -1:
            log(f'Found month at {datestart_idx}')
            if month in months_shortform:
                line.replace(month, months_shortform[month])
            flag_month = True
            break
    if flag_month:
        dateend_idx = line.find('2023')
        if dateend_idx < 0:
            dateend_idx = line.find('2024')
    if dateend_idx > 0:
        datestr = line[datestart_idx:dateend_idx+5]
        log(datestr)
        datestr = convertDate(datestr)
        log(f'{datestr} ||  {date}')

    return datestr

def OCR(image, hours, date):
    log('Entering OCR function...')
    textdata1 = []
    textdata2 = []
    log('Checkpoint 1')
    #months = ['January', 'February', 'March', 'April', 'May', 'June',
    #         'July', 'August', 'September', 'October', 'November', 'December']
    datestr = ''
    log('Checkpoint 2')
    hourstr = hours+" hour" if hours == '1' else hours+" hours"
    log('Checkpoint 3')
    captured_dateline = ''
    log('Checkpoint 4')
    captured_hourline = ''
    log('Checkpoint 5')
    flag_hours1 = False
    flag_hours2 = False
    flag_hours = False
    flag_date1 = False
    flag_date2 = False
    flag_date = False

    captured_dateline = ''
    captured_dateline1 = ''
    captured_dateline2 = ''

    captured_hourline = ''
    captured_hourline1 = ''
    captured_hourline2 = ''

    log('To Gray')
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    log('Processing')
    th3, normed = imgProc(gray_img, outpath)
    log('Processed')
    result1 = reader.readtext(th3)
    result2 = reader.readtext(normed)

    log('Iterating through result 1')
    for (bbox, text, prob) in result1:
            text = cleanup_text(text)
            log(f'Text line: {text}')
            textdata1.append(text)
            #log(f'Textdata: {textdata1}')
    
    log('Iterating through result 2')
    for (bbox, text, prob) in result2:
            text = cleanup_text(text)
            log(f'Text line: {text}')
            textdata2.append(text)
            #log(f'Textdata: {textdata2}')
    
    log(f'Trying OCR on final Gaussian thresholded image')
    for line in textdata1:
        #log(f'Line before Regex: {line}')
        line = line.strip().lower()
        #regex = re.compile('[^A-Za-z0-9\s\r?\n|\r]')
        #line = regex.sub('', line)
        #line = re.sub('[^A-Za-z0-9\s\r?\n|\r]', '', line)
        #line = re.sub('[^a-zA-Z0-9 \n\.]', '', line)
        #log(f'Line after Regex: {line}')
        log(f'Text line[result1]: {line}')

        if flag_hours and flag_date:
            log('Both OK!')
            return flag_hours1, flag_date1, '', ''
        #if hours in line:
        #    flag_hours = True

        if "hour" in line:
            if hours in line:
                flag_hours1 = True
            else:
                log(f'Hours line: {line}')
                captured_hourline1 = line

        #if hours in line:
        #    substr = line[line.find(hours):line.find(hours)+len(hours)+7]
        #    substr = substr.strip()
        #    log(f'Hour str: {hourstr}, substr: {substr}')
        #    if Levenshtein.ratio(hourstr, substr) >= 0.8:
        #        captured_hourline = substr
        #        flag_hours = True
        #    elif Levenshtein.ratio(hourstr, substr) >= 0.7:
        #        captured_hourline = substr

        if ('2023' in line or '2024' in line):
            captured_dateline1 = line
            datestr = extract_datestring_from_ocr_text(line, date)
            #datestr = convertDate(line)
            #if datestr == date:
            if Levenshtein.ratio(datestr, date) >= 0.8:
                flag_date1 = True
    log(f'From Result 1 -> Hours flag: {flag_hours1}, Date flag: {flag_date1}')
    
    log(f'Trying OCR on blurred+divided image')
    for line in textdata2:
        #log(f'Line before Regex: {line}')
        line = line.strip().lower()
        #regex = re.compile('[^A-Za-z0-9\s\r?\n|\r]')
        #line = regex.sub('', line)
        #line = re.sub('[^A-Za-z0-9\s\r?\n|\r]', '', line)
        #line = re.sub('[^a-zA-Z0-9 \n\.]', '', line)
        #log(f'Line after Regex: {line}')
        log(f'Text line[result2]: {line}')

        if flag_hours2 and flag_date2:
            log('Both OK!')
            return flag_hours2, flag_date2, '', ''
        #if hours in line:
        #    flag_hours = True

        if "hour" in line:
            if hours in line:
                flag_hours2 = True
            else:
                log(f'Hours line: {line}')
                captured_hourline2 = line


        #if hours in line:
        #    substr = line[line.find(hours):line.find(hours)+len(hours)+7]
        #    substr = substr.strip()
        #    log(f'Hour str: {hourstr}, substr: {substr}')
        #    if Levenshtein.ratio(hourstr, substr) >= 0.8:
        #        captured_hourline = substr
        #        flag_hours = True
        
        if '2023' in line or '2024' in line:
            captured_dateline2 = line
            datestr = extract_datestring_from_ocr_text(line, date)
            #datestr = convertDate(line)
            #if datestr == date:
            if Levenshtein.ratio(datestr, date) >= 0.8:
                flag_date2 = True
    
    log(f'From Result 2 -> Hours flag: {flag_hours2}, Date flag: {flag_date2}')

    flag_hours = flag_hours1 or flag_hours2
    flag_date = flag_date1 or flag_date2

    log(f'Final Result -> Hours flag: {flag_hours}, Date flag:{flag_date}')
    if flag_hours and flag_date:
        log('Both OK!')
        return flag_hours, flag_date, '', ''
    
    captured_hourline = 'Results: ' + captured_hourline1 + ', ' + captured_hourline2
    captured_dateline = 'Results: ' + captured_dateline1 + ', ' + captured_dateline2

    return flag_hours, flag_date, captured_hourline, captured_dateline

def verifyOCR(fpath, poppler_path, firstname, lastname, hours, date):
    if fpath == '':
        return False, 'not ok: could not process file'
    
    log('Line 403')
    if fpath.endswith('.pdf') or fpath.endswith('.docx'):
        pages = convert_to_img(fpath, poppler_path)
        log('Line 407')
        img = convertBGR2RGB(pages[0])
        log('Line 409')
    else:
        log('Line 411')
        bgr = cv2.imread(fpath)
        log('Line 413')
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        log('Line 415')

    log(f'OCR Verification {firstname} {lastname}: {fpath}')
    flag_hours = False
    flag_date = False
    textdata = ''
    message = 'not ok:'

    log(f'Hours: {hours}, Date: {date}')

    flag_hours, flag_date, hour_text, date_text = OCR(img, hours, date)
    if flag_hours and flag_date:
        return True, "ok"
    
    elif not flag_hours and flag_date:
        message += f' hours ({hour_text})'
    
    elif flag_hours and not flag_date:
        message += f' date ({date_text})'
    
    else:
        message += f' hours({hour_text}) date({date_text})'
    return False, message

def mark_duplicates(partial_df, fulldf, firstname, lastname):
    log(f'Marking duplicates of {firstname} {lastname}...')
    #org = partial_df['HOSTED ORGANIZATION TXT']
    #dup_check = partial_df.duplicated(subset=['First Name', 'Last Name', 'What PD workshop or event did you attend? ', 'Who hosted this event?/What organization sponsored this event?', 'Event start date', 'Event end date'])
    dup_check = partial_df.duplicated(subset=['HOSTED ORGANIZATION TXT', 'What was the start date of this event?', 'What was the end date of this event?'])
    log(f'{dup_check.index}')
    for idx in dup_check.index:
        dup = dup_check[idx]
        row = fulldf.iloc[idx]
        hours = row['How many hours of PD credit are you expecting to receive from your participation?']
        if dup and hours != '1':
            fulldf.at[idx, 'DUP CHECK'] = 'POTENTIAL DUPLICATE'
            log(f'{idx} POTENTIAL DUPLICATE')
        else:
            fulldf.at[idx, 'DUP CHECK'] = 'OK'
            log(f'{idx} OK')
    log(f'Printing partial df \n{partial_df}')
    log(f'Printing duplicate check {dup_check}')

    return fulldf

def main():
    start = datetime.now()
    createLog()

    #accepted_fellows = r'C:\Users\govin\Downloads\accepted_fellows_test.csv'
    accepted_fellows = r'C:\Users\govin\Downloads\accepted_fellows_jan28.csv'

    pdcompletion = r'C:\Users\govin\Downloads\pd_certificates_feb1.csv'

    data1 = pd.read_csv(accepted_fellows)
    df1 = pd.DataFrame(data1)

    data2 = pd.read_csv(pdcompletion)
    df2 = pd.DataFrame(data2)

    scan(df1, df2, start)
    print('Scan done')
    log('Scan done')
    exec_time = datetime.now() - start
    print(exec_time)
    log(f'Execution Time: {exec_time}')


if __name__ == "__main__":
    main()

#url = r"https://www.jotform.com/uploads/hfauland/232128646668061/5823094798634553776/Certificate%20for%20Kathy%20Hartley%201-6-24.pdf"