#!/bin/bash

set -o pipefail -e
export PRELAUNCH_OUT="/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007/prelaunch.out"
exec >"${PRELAUNCH_OUT}"
export PRELAUNCH_ERR="/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007/prelaunch.err"
exec 2>"${PRELAUNCH_ERR}"
echo "Setting up env variables"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games"
export HADOOP_CONF_DIR="/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/etc/hadoop"
export STDOUT_LOGFILE_ENV="/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007/stdout"
export HADOOP_ROOT_LOGGER="INFO,console"
export JAVA_HOME="/home/ccgrid/Desktop/software/jdk1.8.0_161"
export LANG="en_US.UTF-8"
export NM_HOST="ccgrid"
export LD_LIBRARY_PATH="$PWD:$HADOOP_COMMON_HOME/lib/native"
export STDERR_LOGFILE_ENV="/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007/stderr"
export HADOOP_HDFS_HOME="/home/ccgrid/Desktop/hadoop/hadoop-3.0.0"
export LOGNAME="ccgrid"
export JVM_PID="$$"
export PWD="/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/tmp/nm-local-dir/usercache/ccgrid/appcache/application_1575881662489_0045/container_1575881662489_0045_01_000007"
export HADOOP_CLIENT_OPTS=""
export HADOOP_COMMON_HOME="/home/ccgrid/Desktop/hadoop/hadoop-3.0.0"
export LOCAL_DIRS="/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/tmp/nm-local-dir/usercache/ccgrid/appcache/application_1575881662489_0045"
export SHELL="/bin/bash"
export NM_HTTP_PORT="8042"
export LOG_DIRS="/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007"
export NM_AUX_SERVICE_mapreduce_shuffle="AAA0+gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=
"
export NM_PORT="39279"
export USER="ccgrid"
export HADOOP_YARN_HOME="/home/ccgrid/Desktop/hadoop/hadoop-3.0.0"
export CLASSPATH="$PWD:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/etc/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/etc/hadoop/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/lib/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/common/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/common/lib/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/mapreduce/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/mapreduce/lib-examples/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/hdfs/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/hdfs/lib/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/yarn/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/yarn/lib/*::/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/etc/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/etc/hadoop/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/lib/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/common/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/common/lib/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/mapreduce/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/mapreduce/lib-examples/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/hdfs/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/hdfs/lib/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/yarn/*:/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/share/hadoop/yarn/lib/*::job.jar/*:job.jar/classes/:job.jar/lib/*:$PWD/*"
export HADOOP_TOKEN_FILE_LOCATION="/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/tmp/nm-local-dir/usercache/ccgrid/appcache/application_1575881662489_0045/container_1575881662489_0045_01_000007/container_tokens"
export LOCAL_USER_DIRS="/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/tmp/nm-local-dir/usercache/ccgrid/"
export HADOOP_HOME="/home/ccgrid/Desktop/hadoop/hadoop-3.0.0"
export HOME="/home/"
export CONTAINER_ID="container_1575881662489_0045_01_000007"
export MALLOC_ARENA_MAX="4"
echo "Setting up job resources"
ln -sf "/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/tmp/nm-local-dir/usercache/ccgrid/appcache/application_1575881662489_0045/filecache/11/job.jar" "job.jar"
ln -sf "/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/tmp/nm-local-dir/usercache/ccgrid/appcache/application_1575881662489_0045/filecache/13/job.xml" "job.xml"
echo "Copying debugging information"
# Creating copy of launch script
cp "launch_container.sh" "/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007/launch_container.sh"
chmod 640 "/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007/launch_container.sh"
# Determining directory contents
echo "ls -l:" 1>"/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007/directory.info"
ls -l 1>>"/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007/directory.info"
echo "find -L . -maxdepth 5 -ls:" 1>>"/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007/directory.info"
find -L . -maxdepth 5 -ls 1>>"/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007/directory.info"
echo "broken symlinks(find -L . -maxdepth 5 -type l -ls):" 1>>"/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007/directory.info"
find -L . -maxdepth 5 -type l -ls 1>>"/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007/directory.info"
echo "Launching container"
exec /bin/bash -c "$JAVA_HOME/bin/java -Djava.net.preferIPv4Stack=true -Dhadoop.metrics.log.level=WARN   -Xmx820m -Djava.io.tmpdir=$PWD/tmp -Dlog4j.configuration=container-log4j.properties -Dyarn.app.container.log.dir=/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007 -Dyarn.app.container.log.filesize=0 -Dhadoop.root.logger=INFO,CLA -Dhadoop.root.logfile=syslog org.apache.hadoop.mapred.YarnChild 127.0.1.1 40531 attempt_1575881662489_0045_m_000005_0 7 1>/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007/stdout 2>/home/ccgrid/Desktop/hadoop/hadoop-3.0.0/logs/userlogs/application_1575881662489_0045/container_1575881662489_0045_01_000007/stderr "
