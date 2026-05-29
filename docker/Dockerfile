FROM bwu109/motion_prediction@sha256:32e06e6533ce82d267696b8821b9f494d2f508971ab5501e736a65f1fb1ddcc3

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV COOPERSCENE_ROOT=/workspace/CooperScene
WORKDIR /workspace/CooperScene

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
