# Everything in one image: TEMBA, pyBBarolo (3D-Barolo) and SoFiA-2.
#   docker build -t temba -f docker/Dockerfile .
#   docker run --rm -v /path/to/data:/data -v $PWD:/work -w /work temba check survey.yaml
FROM condaforge/miniforge3:latest
WORKDIR /opt/temba
COPY . /opt/temba
RUN conda env create -n temba -f environment.yml && conda clean -afy
SHELL ["conda", "run", "--no-capture-output", "-n", "temba", "/bin/bash", "-c"]
RUN temba install-sofia --prefix /opt/sofia
ENV PATH=/opt/sofia/SoFiA-2:/opt/conda/envs/temba/bin:$PATH
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "temba", "temba"]
CMD ["--help"]
