package KMeans;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import javax.swing.filechooser.FileSystemView;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

import static KMeans.KMeans.Kmeansmain;

public class KMeansResult {
    private static int k = 0;
    private static ArrayList<Cluster> kClusters;

    /*mapper将每个点作为cluster输出是为了在combiner里做提前处理（提前求一次中心点）
     * 因为combiner的输出必须和mapper一样，而在combiner中计算局部中心点时，必须输出用于计算的点的个数以在reducer中进一步算均值
     * 因此如果将点直接作为Instance进行输出，则还需要额外输出点的个数信息，不如直接作为cluster输出
     * */
    public static class KMeansResultMapper extends Mapper<Object, Text, IntWritable, Instance>{
        IntWritable map_key = new IntWritable();
        Instance map_value;

        /*初始化所有的簇*/
        public void setup(Context context) throws IOException{
            k = context.getConfiguration().getInt("k",0);
            kClusters = new ArrayList<Cluster>();
            String cluster_path = context.getConfiguration().get("clusterPath");
            int iter_time = context.getConfiguration().getInt("iter_time", -1);
            //FileSystem hdfs = FileSystem.get(context.getConfiguration());
            FileSystem hdfs = FileSystem.getLocal(context.getConfiguration());
            FSDataInputStream clusterFile = hdfs.open(new Path(cluster_path + "cluster" + iter_time));
            Scanner scan = new Scanner(clusterFile);
            String line = null;
            while(scan.hasNext()){
                line = scan.nextLine();
                Cluster c = new Cluster(line);
                kClusters.add(c);
            }
            clusterFile.close();
            hdfs.close();
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            Instance ins = new Instance(value.toString());
            int id = ins.chooseNearestCluster(kClusters);
            map_key.set(id);
            map_value = ins;
            context.write(map_key, map_value);
        }
    }


    public static void main(String[] args) throws Exception {
        //0:k 1:inputPath 2:iteration times 3:inputfile number
        Kmeansmain(args);
        int fileNum = Integer.parseInt(args[3]);
        int k = Integer.parseInt(args[0]);
        int iter_times = Integer.parseInt(args[2]);
        String inputPath = args[1];
        Configuration conf = new Configuration();
        conf.setInt("k",k);
        conf.setInt("iter_time", iter_times);
        /*获取clusterPath路径*/
        //String os = System.getProperty("os.name");
        File desktopDir = FileSystemView.getFileSystemView().getHomeDirectory();
        String clusterPath = desktopDir.toString();
        String sep = System.getProperty("file.separator");
        clusterPath = clusterPath.concat(sep+"cluster"+sep);
        conf.set("clusterPath",clusterPath);
        Job job = Job.getInstance(conf, "KMeansResult");
        job.setJarByClass(KMeansResult.class);
        job.setMapperClass(KMeansResultMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Instance.class);
        //FileInputFormat.setInputPaths(job, new Path(inputPath));
        /*改为addInputPath*/
        for (int f = 1; f <= fileNum; f++){
            FileInputFormat.addInputPath(job, new Path(inputPath + "Instance" + f + ".txt"));
        }
        FileOutputFormat.setOutputPath(job, new Path("KMeans/temp/result/"));
        System.out.println(job.waitForCompletion(true) ? "success":"failure");
        //job.waitForCompletion(true);
    }
}

