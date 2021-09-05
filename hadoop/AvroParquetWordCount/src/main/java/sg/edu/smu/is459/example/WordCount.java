package sg.edu.smu.is459.example;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.parquet.avro.AvroParquetInputFormat;
import org.apache.hadoop.io.LongWritable;


public class WordCount extends Configured implements Tool {

    public static class TokenizerMapper
            extends Mapper<LongWritable, Post, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(LongWritable key, Post value, Context context
        ) throws IOException, InterruptedException {
            if (value != null) {
                Object object = value.getTopic();
                if(object != null) {
                    String content = object.toString();
                    StringTokenizer itr = new StringTokenizer(content);
                    while (itr.hasMoreTokens()) {
                        word.set(itr.nextToken());
                        context.write(word, one);
                    }
                }
            }
        }
    }

    public static class IntSumReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new WordCount(), args);
        System.exit(res);
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = Job.getInstance(getConf());
        job.setJarByClass(WordCount.class);

        // Set the mapper
        job.setMapperClass(TokenizerMapper.class);
        job.setInputFormatClass(AvroParquetInputFormat.class);
        AvroParquetInputFormat.setInputPaths(job, new Path(args[0]));
        AvroParquetInputFormat.setAvroReadSchema(job, Post.getClassSchema());

        // Intermediate output
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        // Set combiner and reducer
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);

        // The output class is a pair of word and count
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true) ? 0 : 1;
    }
}

