from google_images_download import google_images_download
def main():
    li = ["cat","face","horse","daisy","wall","ball","christmas tree","Eiffel Tower","lalibela building",\
            "axum obelisk","house","computer","office chair","bed","school bag","iphone","water bottle"]   
    response = google_images_download.googleimagesdownload()
    args = {"keywords":",".join(li),"limit":100,"print_urls":True} 
    paths = response.download(args)  
    
if __name__ == "__main__":
    main()