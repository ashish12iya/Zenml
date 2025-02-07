zenml integration install mlflow -y

# Ask and register experiment tracker
read -p "Enter experiment tracker name: " tracker_name
echo "Registering MLflow experiment tracker as '$tracker_name'..."
zenml experiment-tracker register $tracker_name --flavor=mlflow

# Ask and register model deployer
read -p "Enter model deployer name: " deployer_name
echo "Registering MLflow deployer as '$deployer_name'..."
zenml model-deployer register $deployer_name --flavor=mlflow

# Ask and register stack
read -p "Enter stack name: " stack_name
echo "Creating new stack '$stack_name' with registered components..."
zenml stack register $stack_name \
    -o default \
    -a default \
    -e $tracker_name \
    -d $deployer_name

# Set the stack as active
echo "Setting '$stack_name' stack as active..."
zenml stack set $stack_name

echo "Setup completed successfully!" 