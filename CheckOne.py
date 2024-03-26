from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import time
# Set the path to the chromedriver executable
# chromedriver_path = '/path/to/chromedriver'

doSignIn = True
checkCount = 0
times2CheckBeforeEmail = 24
sndToTemp = False
screenshot_path = "screenshot.png"

# retrieve the last checkCount from a file
try:
    with open("checkCount.txt", "r") as file:
        checkCount = int(file.read())
except:
    checkCount = 0


def add_timestamp():

    # Open the screenshot
    img = Image.open(screenshot_path)
    # Prepare to draw on the image
    draw = ImageDraw.Draw(img)

    # Define the timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
#    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Adjust path as needed
#    font = ImageFont.truetype(font_path, 20)

    # Define the text color and position
    text_color = (255, 255, 255)  # White
    text_position = (10, 10)  # Top-left corner

    # Draw the timestamp on the image
    draw.text(text_position, timestamp, fill=text_color)

    # Save the modified image
    img.save(screenshot_path)


def send_email(checkCount, sndToTemp = False):

    global screenshot_path
    sender_address = 'corcompany42@gmail.com'
    sender_pass = 'vgxr wexz yczk cxwx'
    receiver_address = 'gabrielcor@gmail.com'
    tmp_receiver_address = 'corcompany42@gmail.com'

    # Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    if sndToTemp:
        message['To'] = tmp_receiver_address
    else:
        message['To'] = receiver_address

    if checkCount == 0:
        message['Subject'] = 'ATENCION! HAY LUGARES EN LA VISA'
        mail_content = "Hello,\n\nHAY LUGARES EN LA VISA."
    else:
        message['Subject'] = 'Información Actualizada Sobre la Visa'
        mail_content = f"Hello,\n\nCHEQUEE {checkCount} veces y NO HAY LUGARES."
        
    message.attach(MIMEText(mail_content, 'plain'))
    

    part = MIMEBase('application', "octet-stream")
    with open(screenshot_path, 'rb') as file:
        part.set_payload(file.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="screenshot.png"')
    message.attach(part)

    # Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
    session.starttls()  # enable security
    session.login(sender_address, sender_pass)  # login with mail_id and password
    text = message.as_string()
    if sndToTemp:
        session.sendmail(sender_address, tmp_receiver_address, text)
    else:
        session.sendmail(sender_address, receiver_address, text)
    session.quit()



def check_website():
    global checkCount
    driver = webdriver.Chrome() 

    if (doSignIn):
        # Open the webpage
        driver.get('https://ais.usvisa-info.com/en-cl/niv/users/sign_in')

        # Find the email field and fill it with the email address
        email_field = driver.find_element('id','user_email')
        email_field.send_keys('gabrielcor@gmail.com')

        # Find the password field and fill it with the password
        password_field = driver.find_element('id','user_password')
        password_field.send_keys('gnp2juy1kmh_CDN0pch')


        # Instead of finding the input element directly, find the div that represents the checkbox visually
        policy_confirmed_div = driver.find_element(By.CSS_SELECTOR,"div.icheckbox.icheck-item")

        # Click the div to change the checkbox state
        policy_confirmed_div.click()


        # Press the Sign In button
        sign_in_button = driver.find_element(By.NAME,'commit')
        sign_in_button.click()

    # Open the webpage once signed in
    time.sleep(2)
    driver.get('https://ais.usvisa-info.com/en-cl/niv/groups/39834976')
    continue_button = driver.find_element(By.CSS_SELECTOR, "a[href*='schedule/56284522/continue_actions']")
    continue_button.click()

    time.sleep(2)
    schedule_appointment_title = driver.find_element(By.CSS_SELECTOR, "ul.accordion.custom_icons li.accordion-item:first-child a.accordion-title")
    schedule_appointment_title.click()

    time.sleep(2)
    # Assuming 'driver' is your WebDriver instance
    schedule_appointment_button = driver.find_element(By.CSS_SELECTOR, "a.button.small.primary.small-only-expanded")
    schedule_appointment_button.click()


    wait = WebDriverWait(driver, 10)  # Adjust the timeout based on your needs
    consulate_appointment_fields = wait.until(EC.presence_of_element_located((By.ID, "consulate-appointment-fields")))

    time.sleep(2)
    # Step 3: Get the HTML of the section
    consulate_appointment_fields_html = consulate_appointment_fields.get_attribute('outerHTML')
    # A little delay to wait for the html to render on the screen
    # Suspect of being too fast to get the screenshot
    time.sleep(2)

    # Take a screenshot
    driver.save_screenshot(screenshot_path)
    add_timestamp()
    # Now you have the HTML of the section in 'consulate_appointment_fields_html'
    # You can print it or parse it as needed
    search_string = "There are no available appointments at the selected location"
    
    if search_string not in consulate_appointment_fields_html:
        # The string was not found, proceed to send an email
        checkCount = 0
        send_email(checkCount)
    else:
        checkCount += 1
        if checkCount >= times2CheckBeforeEmail:
            send_email(checkCount)
            checkCount = 0
        else:
            send_email(checkCount,True)
    # Close the browser
    driver.quit()



# while True:
    # Place the code to check the website here
check_website()    
# write checkCount to a file
with open("checkCount.txt", "w") as file:
    file.write(str(checkCount))
#    time.sleep(1800)  # Sleep for 30 minutes
# Create a new instance of the Chrome driver
