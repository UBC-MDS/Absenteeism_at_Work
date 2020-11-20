# author: Yiki Su
# date: 2020-11-20

"This script downloads a zip file from the given URL and unzip the file. Save the unzipped file into a local file called data.

Usage: download_data.R --url=<url> 

Options:
----url=<url>   ULR to download the zip file
" -> doc

library(docopt)

opt <- docopt(doc)

main <- function(url) {
  # To download the zip file from the URL
  download.file(url, destfile = "zip")
  
  # To unzip the zip file to the data folder
  unzip("zip", exdir = "data")
}


main(opt$url)

