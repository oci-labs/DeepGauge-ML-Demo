
function iterateShards(kinesis, iter, labels, values) {
  kinesis.getRecords(iter, function(ierr, out) {
    if (ierr) {
      console.log(ierr);
    }
    else {
      out.Records.forEach(function(o, i) {
        var timestamp = new Date(o.ApproximateArrivalTimestamp);
        var jdat = JSON.parse(o.Data);
        labels.push(timestamp.toISOString());
        values.push(jdat.id);
      });

      if (out.MillisBehindLatest == 0) {
        // This is the last shard iterator, we can make the chart now
        new Chart(document.getElementById('line-chart'), {
          type : 'line',
          data : {
            labels : labels,
            datasets : [ { data : values,
                           label : 'Id',
                           fill : false
                         } ]
            },
            options: {
              title : {
                display : true,
                text : 'Values Over Time'
              }
            }
          });
      }
      else {
        // If MillisBehindLatest isn't zero, there is another iterator
	// to go over
	iterateShards(kinesis, { ShardIterator : out.NextShardIterator },
	              labels, values);
      }
    }
  });
}

// read-only-access account with AmazonKinesisReadOnlyAccess policy
var kinesis = new AWS.Kinesis({ region : 'us-east-2',
                                accessKeyId : 'AKIASWEZAXSDQREDVYPI',
                                secretAccessKey : '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' });

var params = {
  StreamName : 'DeepGauge',
  ShardId : 'shardId-000000000000',
  ShardIteratorType : 'AT_TIMESTAMP',
  Timestamp : 1563858000
};

kinesis.getShardIterator(params, function(err, data) {
  if (!err) {
    var labels = [];
    var values = [];
    iterateShards(kinesis, data, labels, values);
  }
});

