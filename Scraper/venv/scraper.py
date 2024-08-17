from multiprocessing.connection import wait
from selenium import webdriver
from selenium.webdriver.common.by import By
import numpy as np
import os
script_dir = os.path.dirname(__file__)
driver = webdriver.Chrome(script_dir+"/chromedriver.exe")

def getAllInternshipLinksFromOnePage(url):
    driver.get(url)
    # Clicking No Thanks on First Time user offer popup
    # Might not appear next time
    try:
        driver.find_element(By.XPATH,'//*[@id="no_thanks"]').click()
    except:
        pass

    # Get all the div tags with category name and company name
    links = driver.find_elements(By.CLASS_NAME,'company')

    # To store the extracted links
    internshipLinks=[]
    for i in range(len(links)):
        # All the 'company' div have two classes: one containing the link of internship;
        # other containing the link for company
        # The first one is captured, which id anchor tag and then its link is captured.
        # Using the code in try because for the advertisements present on webpage, there might 
        # not be companies, therefore no "view_detail_button" attribute
        try:
            internshipLinks.append(links[i].find_element(By.CLASS_NAME, 'view_detail_button').get_attribute('href'))
        except:
            pass
    return internshipLinks

def getAllInternshipLinksFromAllPage():

    # For timing the scraping function
    import time
    start_time = time.time()
    # The page to start scraping
    page = 1
    # Forming the link of first page to scrape
    currlink = 'https://internshala.com/internships/page-' + str(page) + '/'
    # Opening the first page to get the total number of pages
    driver.get(currlink)
    total_pages = driver.find_element(By.XPATH,'//*[@id="total_pages"]').get_attribute('innerHTML')
    total_pages = int(total_pages)
    allLinks=[]

    # looping through each page (first page is again openend)
    while page<=total_pages: # Change value of total_pages to limit the number of pages to be scraped
        currlink = 'https://internshala.com/internships/page-' + str(page) + '/'
        allLinks.append(getAllInternshipLinksFromOnePage(currlink))
        page=page+1
        # Scrolling at end of page
        # next_button = driver.find_element(By.XPATH,'//*[@id="next"]')
        # driver.execute_script('arguments[0].scrollIntoView()', next_button)
    allLinks = sum(allLinks,[])
    allLinks = np.array(allLinks)
    allLinks = allLinks.flatten() # Already flattened before
    print("Total links scraped: ", len(allLinks))
    # driver.close()
    end_time = time.time()
    print("Scraping time: ", str(end_time-start_time) + " Seconds")
    return allLinks

# Saves all the links scraped to a CSV with current time and returns the original allLinks
def saveAllLinks(allLinks):
    import csv
    from datetime import datetime
    data = allLinks.reshape((-1,1))
    currentTime = datetime.now().strftime("%d%m%y_%H%M%S")
    file_name = script_dir+"/all_links_"+currentTime+".csv"
    file = open(file_name,'w',newline='')
    writer = csv.writer(file)
    writer.writerows(data)
    file.close()
    return allLinks
# saveAllLinks(getAllInternshipLinksFromAllPage())

# Enter only list of elements
def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele.text.strip()+","

        # return string
    return str1

# Returns a dictonary of all the details of Internship present on the link.
def getInternshipData(link):
    try:
        driver.get(link)
        # Clicking No Thanks on First Time user offer popup
        # Might not appear next time
        try:
            driver.find_element(By.XPATH, '//*[@id="no_thanks"]').click()
        except:
            pass

        category = driver.find_element(By.CLASS_NAME, 'profile_on_detail_page').get_attribute('innerHTML').strip()
        company = driver.find_element(By.CLASS_NAME, 'company_and_premium').find_element(By.TAG_NAME,'a').get_attribute('innerHTML').strip()
        location = listToString(driver.find_element(By.XPATH, '//*[@id="location_names"]/span').text.strip()) # To make sure there is ',' at end. Easy to preprocess

        date_Duration_Stipend_apply = driver.find_elements(By.CLASS_NAME, 'other_detail_item')
        start_date=date_Duration_Stipend_apply[0].find_element(By.CLASS_NAME,'item_body').text.strip()
        duration = date_Duration_Stipend_apply[1].find_element(By.CLASS_NAME,'item_body').text.strip()
        stipend = date_Duration_Stipend_apply[2].find_element(By.CLASS_NAME,'item_body').text.strip()
        apply_by = date_Duration_Stipend_apply[3].find_element(By.CLASS_NAME,'item_body').text.strip()

        applicants = driver.find_element(By.CLASS_NAME, 'applications_message_container').text.strip()
        try:
            applicants = applicants.split()[0]
            applicants = int(applicants)
        except:
            applicants = 0

        skills_required = None
        # Skills required is found by first finding the element with heading 'skill'. If it is present, then the element next to it is picked
        # and skills required is extracted
        try:
            elements = driver.find_elements(By.CLASS_NAME, 'section_heading.heading_5_5')
            for element in elements:
                if element.text.strip().startswith('Skill'):
                    skills_required_container = element.find_element(By.XPATH, './following-sibling::div')
                    skills_required_container = skills_required_container.find_elements(By.CLASS_NAME, 'round_tabs')
                    skills_required = listToString(skills_required_container)
                    break
        except Exception as err:
            pass

        # Perks is found by first finding the element with heading 'Perks'. If it is present, then the element next to it is picked
        # and perks is extracted
        perks = None
        try:
            elements = driver.find_elements(By.CLASS_NAME, 'section_heading.heading_5_5')
            for element in elements:
                if element.text=='Perks':
                    perks_container = element.find_element(By.XPATH,'./following-sibling::div')
                    perks_container = perks_container.find_elements(By.CLASS_NAME, 'round_tabs')
                    perks = listToString(perks_container)
                    break
        except Exception as err:
            pass

        openings = None
        try:
            elements = driver.find_elements(By.CLASS_NAME, 'section_heading.heading_5_5')
            for element in elements:
                if element.text=='Number of openings':
                    openings_container = element.find_element(By.XPATH,'./following-sibling::div').text
                    openings = int(openings_container)
                    break
        except Exception as err:
            pass
        return {'category':category, 'company':company, 'location':location, 'start_date':start_date, 'duration':duration,
            'stipend':stipend, 'apply_by':apply_by,'applicants':applicants,'skills_required':skills_required,
            'perks':perks,'openings':openings,'link':link}
    except Exception as e:
        print(e)
        return {'category': None, 'company': None, 'location': None, 'start_date': None,
                'duration': None,
                'stipend': None, 'apply_by': None, 'applicants': None,
                'skills_required': None,
                'perks': None, 'openings': None, 'link': link}

def getInternshipDataOfAllLinks(allLinks):
    allData = []
    # For timing the scraping function
    import time
    start_time = time.time()
    i=1
    for link in allLinks:
        allData.append(getInternshipData(link))
        if i%40==0:
            print("Total internships scraped: ",str(i))
        i=i+1
    end_time = time.time()
    print("Total internships scraped: ",len(allLinks))
    print("Scraping time: ", str(end_time - start_time) + " Seconds")
    return allData

def saveAllInternshipData(listOfDict):
    import csv
    from datetime import datetime
    currentTime = datetime.now().strftime("%d%m%y_%H%M%S")
    file_name = script_dir+"/all_internships_"+currentTime+".csv"
    keys=listOfDict[0].keys()
    with open(file_name,'w',encoding="utf-8",newline='') as file:
        dictWriter = csv.DictWriter(file,keys)
        dictWriter.writeheader()
        dictWriter.writerows(listOfDict)

saved_links = saveAllLinks(getAllInternshipLinksFromAllPage())
all_internship_data = getInternshipDataOfAllLinks(saved_links)
saveAllInternshipData(all_internship_data)


driver.close()
print("Finished")

