FROM rocker/tidyverse

USER root

RUN apt-get update --fix-missing 

# install python3
RUN apt-get install -y \
		python3-pip \
		python3-dev 
		

# install R packages
RUN apt-get update -qq && install2.r --error \
    --deps TRUE \
    knitr \
    feather \
    ggcorrplot \
    ggthemes \
    reticulate \
    docopt 
    
# install the anaconda 

# install python packages
    

    
   