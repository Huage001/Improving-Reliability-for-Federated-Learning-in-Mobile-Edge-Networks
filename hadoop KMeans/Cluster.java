package KMeans;

import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;


public class Cluster implements WritableComparable<Cluster> {
    private int clusterID;
    private long numOfPoints;
    private Instance center;

    public Cluster() {
        clusterID = -1;
        numOfPoints = 0;
        center = new Instance();
    }

    public Cluster(int id, long num, Instance ins){
        clusterID = id;
        numOfPoints = num;
        center = ins;
    }

    //@param s <id>\t<id>,<numOfPoints>,<Instance>
    public Cluster(String s){
        String[] info = s.split("[\t,]",4); // 仅处理前三个\t和逗号，保持坐标向量仍为字符串
        clusterID = Integer.parseInt(info[1]);
        numOfPoints = Long.parseLong(info[2]);
        center = new Instance(info[3]);
    }

    public Cluster add(Cluster cluster){
        if(clusterID != cluster.getClusterID()) System.out.printf("簇序号不同，不能相加！");
        Cluster sum = new Cluster(clusterID, numOfPoints+cluster.getNumOfPoints(),center.add(cluster.getCenter()));
        return sum;
    }

    public int getClusterID() {
        return clusterID;
    }

    public long getNumOfPoints() {
        return numOfPoints;
    }

    public Instance getCenter() {
        return center;
    }

    public void setClusterID(int clusterID) {
        this.clusterID = clusterID;
    }

    public void setNumOfPoints(long numOfPoints) {
        this.numOfPoints = numOfPoints;
    }

    public void setCenter(Instance center) {
        this.center = center;
    }

    @Override
    public String toString() {
        return Integer.toString(clusterID) + "," + Long.toString(numOfPoints) + "," + center.toString();
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(clusterID);
        out.writeLong(numOfPoints);
        center.write(out);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        clusterID = in.readInt();
        numOfPoints = in.readLong();
        center.readFields(in);
    }

    @Override
    public int compareTo(Cluster c) {
        if(clusterID < c.getClusterID()) return -1;
        else if (clusterID == c.getClusterID()) return 0;
        else return 1;
    }
}
