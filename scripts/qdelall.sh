
#! /bin/sh
# If two arguments are given, they will be treated as a inclusive range of job ids to delete from the queue
if [ "$#" -eq 2 ]; then
        echo "Deleting Job IDs $1 to $2"
            for i in $(seq $1 $2); do
                        echo $i
                            done
                        else
                            # Otherwise, for each incomplete job owned by the user, delete it from the job queue
                            #
                            # Find all the jobs owned by the user
                            # qstat -u $USERNAME
                            #
                            # The output has the following column format:
                            # Job ID               Username    Queue    Jobname          SessID NDS   TSK    Memory Time  S   Time
                            # $1                   $2          $3       $4               $5     $6    $7     $8     $9    $10 $11
                            #
                            #
                            # Print the Job ID if the state is anything other than complete
                            # The Username column is checked to skip any header lines
                            # awk -v username="$USERNAME" '$2 == username && $10 != "C" { print $1 }'
                            #
                            # Trim everything after the number part of the Job ID
                            # cut -d . -f 1
                            #
                            # Delete the job id from the queue
                            # qdel $job_id
                                USERNAME=$(whoami)
                                    echo "Deleting all incomplete jobs owned by $USERNAME"
                                        for job_id in $(qstat -u $USERNAME | awk -v username="$USERNAME" '$2 == username && $10 != "C" { print $1 }' | cut -d . -f 1); do
                                                    qdel $job_id
                                                        done
                                                    fi

