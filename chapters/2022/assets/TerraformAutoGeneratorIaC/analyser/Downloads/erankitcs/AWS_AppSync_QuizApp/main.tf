resource "aws_iam_role" "graphqlapirole" {
  name = "graphql_api_role"

  assume_role_policy = <<POLICY
{
    "Version": "2012-10-17",
    "Statement": [
        {
        "Effect": "Allow",
        "Principal": {
            "Service": "appsync.amazonaws.com"
        },
        "Action": "sts:AssumeRole"
        }
    ]
}
POLICY
}

resource "aws_iam_role_policy_attachment" "cw_log_policy_attach" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppSyncPushToCloudWatchLogs"
  role       = aws_iam_role.graphqlapirole.name
}

resource "aws_appsync_graphql_api" "quiz_gql_api" {
  authentication_type = "API_KEY"
  name                = "QuizApp API"
  schema = <<EOF
type Question {
  id: ID!
  text: String!
  explanation: String
  answers: [AWSJSON]
}

type PaginatedQuestions {
  nextToken: String
  items: [Question]
}

type Query {
  getQuestion(id: ID!): Question
  listQuestions(limit: Int, nextToken: String): PaginatedQuestions
}

input CreateQuestionInput {
  text: String!
  explanation: String
  answers: [AWSJSON]
}

type Mutation {
  createQuestion(input: CreateQuestionInput!): Question
}
EOF
  log_config {
    cloudwatch_logs_role_arn = aws_iam_role.graphqlapirole.arn
    field_log_level          = "ALL"
  }
}

resource "aws_appsync_api_key" "quiz_gql_api_key" {
  api_id  = aws_appsync_graphql_api.quiz_gql_api.id
  #expires = "2018-05-03T04:00:00Z" default is 7 days.
}