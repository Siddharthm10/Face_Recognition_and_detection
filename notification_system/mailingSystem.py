def Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES):
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('./notification_system/creds/token/token.pickle'):
        with open('./notification_system/creds/token/token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('./notification_system/creds/token/token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build(API_NAME, API_VERSION, credentials=creds)
    return service




if __name__ == '__main__':

    from email import message
    import os
    import base64
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    import mimetypes
    import pickle
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request


    CLIENT_SECRET_FILE = './notification_system/creds/credentials.json'
    API_NAME ='gmail'
    API_VERSION = 'v1'
    SCOPES = ["https://mail.google.com/"]



    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

    file_attachments = ['./attachments/selfie1.jpg']

    emailMsg = 'One File Attached'

    mimeMessage = MIMEMultipart()
    mimeMessage['to'] = 'abhijeet130499@gmail.com'
    mimeMessage['subject'] = 'You got files'
    mimeMessage.attach(MIMEText(emailMsg,'plain'))
    #attach file
    for attachment in file_attachments:
        content_type, encoding = mimeMessage.guess_type(attachment)
        main_type, sub_type = content_type.split('/', 1)
        file_name = os.path.basename(attachment)

        f = open(attachment, 'rb')

        myfile = MIMEBase(main_type, sub_type)
        myfile.set_payload(f.read())
        myfile.add_header('content-Disposition', 'attachment', file_name)
        
        f.close()

        mimeMessage.attach(myfile)

    raw_string = base64.urlsafe_b64decode(mimeMessage.as_bytes()).decode

    service.users().message().send(
        userId='me',
        body={'raw':raw_string}).execute()

    print(message)