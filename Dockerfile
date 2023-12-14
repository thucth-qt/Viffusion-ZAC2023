FROM thtuancs/zac2023:env_nocode

RUN apt-get clean
RUN rm -r /tmp/*
RUN rm -r ~/.cache/pip
ADD zac2023/ /code
