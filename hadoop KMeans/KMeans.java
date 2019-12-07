package KMeans;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.MRJobConfig;

import javax.swing.filechooser.FileSystemView;
import java.io.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.Random;
import java.util.Scanner;

public class KMeans {
    private static int k = 0;
    private static ArrayList<Cluster> kClusters;

    /*mapper将每个点作为cluster输出是为了在combiner里做提前处理（提前求一次中心点）
    * 因为combiner的输出必须和mapper一样，而在combiner中计算局部中心点时，必须输出用于计算的点的个数以在reducer中进一步算均值
    * 因此如果将点直接作为Instance进行输出，则还需要额外输出点的个数信息，不如直接作为cluster输出
    * */
    public static class KMeansMapper extends Mapper<Object, Text, IntWritable, Cluster>{
        IntWritable map_key = new IntWritable();
        Cluster map_value;

        /*初始化所有的簇*/
        public void setup(Context context) throws IOException{
            k = context.getConfiguration().getInt("k",0);
            kClusters = new ArrayList<Cluster>();
            String cluster_path = context.getConfiguration().get("clusterPath");
            int iter_time = context.getConfiguration().getInt("iter_time", -1);
            //FileSystem hdfs = FileSystem.get(context.getConfiguration());
            FileSystem hdfs = FileSystem.getLocal(context.getConfiguration());
            FSDataInputStream clusterFile = hdfs.open(new Path(cluster_path + "cluster" + (iter_time-1)));
            Scanner scan = new Scanner(clusterFile);
            String line = null;
            while(scan.hasNext()){
                line = scan.nextLine();
                Cluster c = new Cluster(line);
                kClusters.add(c);
            }
            clusterFile.close();
            hdfs.close();
            /*输出时间戳信息*/
            InputSplit is = context.getInputSplit();
            //String splitId = MD5Hash.digest(is.toString()).toString();
            String splitId = is.toString();
            Date dNow = new Date();
            java.text.SimpleDateFormat time = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss SSS");
            System.out.println("Time: " + time.format(dNow) + " startMap " + splitId);
            assert (kClusters.size() == k);
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            Instance ins = new Instance(value.toString());
            int id = ins.chooseNearestCluster(kClusters);
            assert id != -1;
            Cluster cluster = new Cluster(id,1,ins);
            map_key.set(id);
            map_value = cluster;
            context.write(map_key, map_value);
        }

        public void cleanup(Context context) throws IOException{
            InputSplit is = context.getInputSplit();
            //String splitId = MD5Hash.digest(is.toString()).toString();
            String splitId = is.toString();
            Date dNow = new Date();
            java.text.SimpleDateFormat time = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss SSS");
            System.out.println("Time: " + time.format(dNow) + " endMap " + splitId);
            assert (kClusters.size() == k);
        }
    }

    public static class KMeansCombiner extends Reducer<IntWritable, Cluster, IntWritable, Cluster> {
        IntWritable reduce_key = new IntWritable();
        Cluster reduce_value = new Cluster();

        public void reduce(IntWritable key, Iterable<Cluster> value, Context context)
                throws IOException, InterruptedException{
            long total_num = 0;
            long num = 0;
            Instance ins = new Instance();
            for(Cluster cluster:value){
                num = cluster.getNumOfPoints();
                /*第一次循环*/
                if(total_num == 0){
                    ins = cluster.getCenter().multiply(num);
                }
                else{
                    ins = ins.add(cluster.getCenter().multiply(num));
                }
                total_num += cluster.getNumOfPoints();
            }
            ins = ins.divide(total_num);
            reduce_value.setClusterID(key.get());
            reduce_value.setNumOfPoints(total_num);
            reduce_value.setCenter(ins);
            reduce_key.set(key.get());
            context.write(reduce_key,reduce_value);
        }

    }

    public static class KMeansReducer extends Reducer<IntWritable, Cluster, IntWritable, Cluster> {
        IntWritable reduce_key = new IntWritable();
        Cluster reduce_value = new Cluster();

        public void setup(Context context) throws IOException{
            //OutputCommitter is = context.getOutputCommitter();
            String s = MRJobConfig.TASK_ID;
            //String splitId = MD5Hash.digest(is.toString()).toString();
            //String splitId = is.toString();
            Date dNow = new Date();
            java.text.SimpleDateFormat time = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss SSS");
            System.out.println("Time: " + time.format(dNow) + " startReduce " + s);
        }

        public void reduce(IntWritable key, Iterable<Cluster> value, Context context)
                throws IOException, InterruptedException{
            long total_num = 0;
            long num = 0;
            Instance ins = new Instance();
            for(Cluster cluster:value){
                num = cluster.getNumOfPoints();
                /*第一次循环*/
                if(total_num == 0){
                    ins = cluster.getCenter().multiply(num);
                }
                else{
                    ins = ins.add(cluster.getCenter().multiply(num));
                }
                total_num += cluster.getNumOfPoints();
            }
            ins = ins.divide(total_num);
            reduce_value.setClusterID(key.get());
            reduce_value.setNumOfPoints(total_num);
            reduce_value.setCenter(ins);
            reduce_key.set(key.get());
            //context.write(reduce_key,reduce_value);
            /*将文件写入本地*/
            Configuration conf = context.getConfiguration();
            String cluster_path = conf.get("clusterPath");
            int iter_time = conf.getInt("iter_time", -1);
            //FileSystem hdfs = FileSystem.getLocal(conf);
            //FSDataOutputStream cluster_output = hdfs.create(new Path(cluster_path+"cluster"+iter_time), true);
            FileOutputStream cluster_output = new FileOutputStream(cluster_path + "cluster" + iter_time, true);
            BufferedWriter out = new BufferedWriter(new OutputStreamWriter(cluster_output,"UTF-8"));
            //OutputStreamWriter out = new OutputStreamWriter(cluster_output, "UTF-8");
            String out_line;
            out_line = reduce_key + "\t" + reduce_value.toString();
            out.write(out_line);
            out.newLine();
            out.flush();
            out.close();
            cluster_output.close();
        }

        public void cleanup(Context context) throws IOException{
            //OutputCommitter is = context.getOutputCommitter();
            //String splitId = MD5Hash.digest(is.toString()).toString();
            //String splitId = is.toString();
            String s = MRJobConfig.TASK_ID;
            Date dNow = new Date();
            java.text.SimpleDateFormat time = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss SSS");
            System.out.println("Time: " + time.format(dNow) + " endReduce " + s);
        }

    }


    public static void Kmeansmain(String[] args) throws Exception {
        //0:k 1:inputPath 2:iteration times 3:inputfile number
        String inputPath = args[1];
        int iter_times = Integer.parseInt(args[2]);
        Configuration conf = new Configuration();
        conf.setInt("k",Integer.parseInt(args[0]));
        /*获取clusterPath路径*/
        //String os = System.getProperty("os.name");
        File desktopDir = FileSystemView.getFileSystemView().getHomeDirectory();
        String clusterPath = desktopDir.toString();
        String sep = System.getProperty("file.separator");
        clusterPath = clusterPath.concat(sep+"cluster"+sep);
        conf.set("clusterPath",clusterPath);
        /*生成本地Cluster0文件*/
        FileSystem hdfs = FileSystem.get(conf);
        FSDataInputStream instance_input = hdfs.open(new Path(args[1] + "Instance1.txt"));
        FileOutputStream cluster_output = new FileOutputStream(clusterPath+"cluster0");
        BufferedReader in = new BufferedReader(new InputStreamReader(instance_input, "UTF-8"));
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(cluster_output,"UTF-8"));
        //FSDataOutputStream cluster_output = hdfs.create(new Path(conf.get("clusterPath")+"cluster0"));
        //FSDataInputStream instance_input = hdfs.open(new Path(args[1] + "Instance1.txt"));
        //BufferedWriter out = new BufferedWriter(new OutputStreamWriter(cluster_output,"UTF-8"));
        //BufferedReader in = new BufferedReader(new InputStreamReader(instance_input, "UTF-8"));
        /*读入实例点，并随机选出初始簇的实例点*/
        ArrayList<Instance> ori_cluster = new ArrayList<Instance>();
        String line = null;
        Instance ins;
        int k = Integer.parseInt(args[0]);
        Random r = new Random();
        while((line = in.readLine()) != null){
            ins = new Instance(line);
            if(ori_cluster.size() < k){
                ori_cluster.add(ins);
            }
            else{
                double random = r.nextFloat();
                if(random < 1.0/(1+k)){
                    int num_covered = r.nextInt(k);
                    ori_cluster.set(num_covered, ins);
                }
            }
        }
        in.close();
        hdfs.close();
        /*将随机出的实例点写入cluster0*/
        String out_line;
        for(int id = 0; id < k; id++){
            out_line = Integer.toString(id) + "\t" + Integer.toString(id) + "," + Integer.toString(0) + "," + ori_cluster.get(id).toString();
            out.write(out_line);
            out.newLine();
            out.flush();
        }
        out.close();
        cluster_output.close();
        /*改为addInputPath后多输入一个文件数*/
        int fileNum = Integer.parseInt(args[3]);
        /*循环执行*/
        for(int i = 1; i <= iter_times; i++){
            conf.set("clusterPath",clusterPath);
            conf.setInt("iter_time", i);
            Job job = Job.getInstance(conf, "KMeans");
            job.setNumReduceTasks(1); // 将reduce设置为1
            job.setJarByClass(KMeans.class);
            job.setMapperClass(KMeansMapper.class);
            job.setReducerClass(KMeansReducer.class);
            job.setCombinerClass(KMeansCombiner.class);
            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(Cluster.class);
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(Cluster.class);
            //FileInputFormat.setInputPaths(job, new Path(inputPath));
            /*改为addInputPath*/
            for (int f = 1; f <= fileNum; f++){
                FileInputFormat.addInputPath(job, new Path(inputPath + "Instance" + f + ".txt"));
            }
            /*不设置OutputPath，输出到本地*/
            FileOutputFormat.setOutputPath(job, new Path("KMeans/temp/"+ "cluster" + (i + 1)));
            //System.exit(job.waitForCompletion(true) ? 0 : 1);
            System.out.println(job.waitForCompletion(true) ? "success":"failure");
        }

    }
}
